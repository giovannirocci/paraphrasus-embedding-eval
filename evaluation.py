import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import json
import os
from tqdm import tqdm

from calibration import threshold_learning, classifier_learning
from embedding import compute_scores, load_embedder


def compute_all_datasets(model_id: str, datasets_dir: str, outdir: str):
    """
    Compute scores for all datasets in the specified directory.
    """
    os.makedirs(outdir, exist_ok=True)
    model = load_embedder(model_id)

    results = {}
    print("Computing similarity scores...")
    for ds_file in tqdm(os.listdir(datasets_dir), total=len(os.listdir(datasets_dir))):
        if args.paraphrasus_consistent:
            # Only use datasets from original Paraphrasus paper
            if "tapaco_paraphrases" in ds_file:
                continue
        if ds_file.endswith(".json"):
            ds_path = os.path.join(datasets_dir, ds_file)
            ds_name = ds_file.replace(".json", "")
            results[ds_name] = compute_scores(model, model_id, ds_path, args.method, multi_eval=True)
    return results


def loo_eval(datasets, metric: str, calibration: str):
    """
    Leave-One-Out evaluation with calibration.
    """
    if metric == "auc":
        raise ValueError("AUC metric is not supported with calibration methods.")

    results = {}
    print(f"Performing Leave-One-Out evaluation with {calibration} calibration...")
    for held_out in datasets:
        train_labels, train_scores, train_diffs = [], [], []

        for ds_name, data in datasets.items():
            if ds_name == held_out:
                continue
            train_labels.extend(data["labels"])
            train_scores.extend(data["scores"])
            train_diffs.extend(data["diffs"])

        # ---- Calibration ----
        if calibration == "threshold":
            best_thr, _ = threshold_learning(np.array(train_scores), np.array(train_labels))
        elif calibration == "classifier":
            X_train = np.asarray(train_diffs, dtype=np.float32)
            y_train = np.asarray(train_labels)
            clf = classifier_learning(X_train, y_train)
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")

        # ---- Evaluation on held-out ----
        print(f"Evaluating on held-out dataset: {held_out}")
        held_out_data = datasets[held_out]
        held_labels = np.asarray(held_out_data["labels"])

        if calibration == "classifier":
            X_test = np.asarray(held_out_data["diffs"], dtype=np.float32)
            preds = clf.predict(X_test)
        else:
            scores = np.asarray(held_out_data["scores"], dtype=np.float32)
            preds = (scores > best_thr).astype(np.int32)

        # ---- Metrics ----
        if metric == "f1":
            results[held_out] = f1_score(held_labels, preds, zero_division=1)
        elif metric == "error":
            acc = accuracy_score(held_labels, preds)
            results[held_out] = 1 - acc
        else:
            raise ValueError(f"Unknown metric {metric}.")

    # ---- Aggregate results ----
    classify, minimize, maximize = [], [], []
    for k, v in datasets.items():
        if v["goal"] == "classify":
            classify.append(results[k])
        elif v["goal"] == "minimize":
            minimize.append(results[k])
        elif v["goal"] == "maximize":
            maximize.append(results[k])

    def aggregate(group):
        if not group:
            return None
        return float(np.mean(group))

    results["overall_classify"] = aggregate(classify)
    results["overall_minimize"] = aggregate(minimize)
    results["overall_maximize"] = aggregate(maximize)

    results["overall_mean"] = aggregate([results["overall_classify"], results["overall_minimize"], results["overall_maximize"]])

    return results


def single_eval(model_id, ds_path, metric: str, calibration: str):
    """
    Single dataset evaluation.
    """
    from sklearn.model_selection import train_test_split

    model = load_embedder(model_id)

    data = compute_scores(model, model_id, ds_path, args.method, multi_eval=False)
    ds_name = os.path.basename(ds_path).replace(".json", "")

    if metric == "auc":
        if calibration:
            raise Warning("AUC metric is not supported with calibration methods.")
        auc = roc_auc_score(data["labels"], data["scores"])
        return {f"{ds_name}_{metric}": auc}
    else:
        results = {}
        y = np.asarray(data["labels"], dtype=np.int32)
        
        if calibration == "threshold":
            X = np.asarray(data["scores"], dtype=np.float32)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            best_thr, _ = threshold_learning(X_train, y_train)
            # Eval
            preds = (X_test > best_thr).astype(np.int32)

        elif calibration == "classifier":
            X = np.asarray(data["diffs"], dtype=np.float32)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = classifier_learning(X_train, y_train)
            # Eval
            preds = clf.predict(X_test)
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")

        if metric == "f1":
            results[metric] = f1_score(y_test, preds, zero_division=1)
        elif metric == "error":
            acc = accuracy_score(y_test, preds)
            results[metric] = 1 - acc
        else:
            raise ValueError(f"Unknown metric {metric}.")

    return results


def main(model: str, metric: str, calibration: str, datasets_dir: str, outdir: str, single: bool = False):
    if not single:
        datasets = compute_all_datasets(model, datasets_dir, outdir)

    results = {}
    if args.full:
        if single:
            raise ValueError("Full evaluation is not supported for single dataset evaluation.")
        for met in ["auc", "f1", "error"]:
            if met == "auc":
                # Use similarity scores from the base datasets
                all_labels = np.concatenate([data["labels"] for data in datasets.values()])
                all_scores = np.concatenate([data["scores"] for data in datasets.values()])
                print("Computing overall AUC...")
                auc = roc_auc_score(all_labels, all_scores)
                results[f"overall_{met}"] = auc

            else:
                # ---------- Threshold calibration ----------
                print(f"\nRunning {met.upper()} eval with THRESHOLD calibration")
                threshold_results = loo_eval(datasets, met, "threshold")
                results[f"threshold_{met}"] = threshold_results

                # ---------- Classifier calibration ----------
                print(f"\nRunning {met.upper()} eval with CLASSIFIER calibration")
                classifier_results = loo_eval(datasets, met, "classifier")
                results[f"classifier_{met}"] = classifier_results

        if args.paraphrasus_consistent:
            results_path = os.path.join(outdir, f"{model.replace('/', '_')}_comparable_full_results.json")
        else:
            results_path = os.path.join(outdir, f"{model.replace('/', '_')}_{args.method}_full_results.json")

    elif single:
        print(f"Evaluating single dataset: {args.ds_path.split('/')[-1].replace('.json','')}")
        results = single_eval(model, args.ds_path, metric, calibration)
        return results
    
    else:
        if calibration is None and metric == "auc":
            print("Computing overall AUC...")
            all_labels = np.concatenate([data["labels"] for data in datasets.values()])
            all_scores = np.concatenate([data["scores"] for data in datasets.values()])
            auc = roc_auc_score(all_labels, all_scores)
            results = {"overall_auc": auc}
        else:
            results = loo_eval(datasets, metric, calibration)

        results_path = os.path.join(outdir,
                                    f"{model.replace('/', '_')}_{metric}_{calibration if calibration else ''}"
                                    f"_{args.method if calibration == 'classifier' else ''}_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path on HuggingFace Hub or Sbert models")
    parser.add_argument("--metric", choices=["auc", "error", "f1"], help="Evaluation metric")
    parser.add_argument("--calibration", choices=["threshold", "classifier"], help="Calibration method")
    parser.add_argument("--datasets_dir", default="datasets_no_results", help="Directory containing datasets in JSON format")
    parser.add_argument("--outdir", default="embedding_benchmarks", help="Output directory for results")
    parser.add_argument("--full", action="store_true", help="Evaluate on all metrics and all calibration methods")
    parser.add_argument("--paraphrasus_consistent", action="store_true",
                        help="Use only datasets from original Paraphrasus paper, to get comparable results.")
    parser.add_argument("--method", choices=["elementwise_diff", "multiplication", "sum"], default="elementwise_diff")
    parser.add_argument("--single_dataset", help="Evaluate on a single dataset", action="store_true")
    parser.add_argument("--ds_path", help="Path to the single dataset JSON file")
    args = parser.parse_args()

    if args.metric in ["error", "f1"] and args.calibration is None and not args.full:
        raise ValueError("Metric 'error' or 'f1' requires a calibration method ('--calibration') or '--full' flag.")
    
    if args.single_dataset and not args.ds_path:
        raise ValueError("Please provide the path to the single dataset using '--ds_path'.")
    
    os.makedirs(args.outdir, exist_ok=True)

    main(args.model, args.metric, args.calibration, args.datasets_dir, args.outdir, args.single_dataset)