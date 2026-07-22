import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import json
import os, random
from tqdm import tqdm

from calibration import threshold_learning, classifier_learning
from embedding import compute_scores, load_embedder

random.seed(42)

def compute_all_datasets(model_id: str, datasets_dir: str, clf_method: str, paraphrasus_consistent: bool = False, prompt: str = None):
    """
    Compute scores for all datasets in the specified directory.
    """
    model = load_embedder(model_id)

    results = {}
    print("Computing similarity scores...")
    for ds_file in tqdm(os.listdir(datasets_dir), total=len(os.listdir(datasets_dir))):
        if paraphrasus_consistent:
            # Only use datasets from original Paraphrasus paper
            if "tapaco_paraphrases" in ds_file:
                continue
        if ds_file.endswith(".json"):
            ds_path = os.path.join(datasets_dir, ds_file)
            ds_name = ds_file.replace(".json", "")
            results[ds_name] = compute_scores(model, model_id, ds_path, clf_method, prompt=prompt)
    return results


def loo_eval(datasets: dict, metric: str, calibration: str, held_out_dataset: str = None, calibration_sample_size: int = 500):
    """
    Leave-One-Out evaluation with calibration.
    Args:
        datasets: dictionary containing scores, diffs, labels, goal for each dataset
        metric: evaluation metric ("auc", "f1", "error")
        calibration: calibration method ("threshold", "classifier")
        held_out_dataset: specific dataset to hold out (optional). If None, performs LOO on all datasets.
        calibration_sample_size: number of samples per dataset to use for calibration.
    Returns:
        results: dict with evaluation results
    """
    if metric == "auc":
        raise ValueError("AUC metric is not supported with calibration methods.")

    results = {}
    if held_out_dataset is not None:
        if held_out_dataset not in datasets:
            raise ValueError(f"Specified held-out dataset '{held_out_dataset}' not found in datasets.")
        held_out_datasets = [held_out_dataset]
        print(f"Performing evaluation with {held_out_dataset} as held-out dataset using {calibration} calibration...")
    else:
        held_out_datasets = datasets.keys()
        print(f"Performing Leave-One-Out evaluation with {calibration} calibration...")
    
    for held_out in held_out_datasets:
        train_labels, train_scores, train_diffs = [], [], []

        for ds_name, data in datasets.items():
            if ds_name == held_out:
                continue

            if len(data["labels"]) > calibration_sample_size:
                idxs = random.sample(range(len(data["labels"])), calibration_sample_size)
                labels = [data["labels"][i] for i in idxs]
                scores = [data["scores"][i] for i in idxs]
                diffs = [data["diffs"][i] for i in idxs]
                print(f"Sampled {calibration_sample_size} pairs from {ds_name} for training (avoid overfitting).")
            else:
                labels = data["labels"]
                scores = data["scores"]
                diffs = data["diffs"]

            train_labels.extend(labels)
            train_scores.extend(scores)
            train_diffs.extend(diffs)

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
        y_test = np.asarray(held_out_data["labels"])

        if calibration == "classifier":
            X_test = np.asarray(held_out_data["diffs"], dtype=np.float32)
            preds = clf.predict(X_test)
        else:
            scores = np.asarray(held_out_data["scores"], dtype=np.float32)
            preds = (scores > best_thr).astype(np.int32)

        # ---- Metrics ----
        if metric == "f1":
            results[held_out] = f1_score(y_test, preds, zero_division=1)
        elif metric == "error":
            acc = accuracy_score(y_test, preds)
            results[held_out] = 1 - acc
        else:
            raise ValueError(f"Unknown metric {metric}.")

    # ---- Aggregate results ----
    if not held_out_dataset:
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


def single_eval(model_id, ds_path, metric: str, calibration: str, prompt: str = None):
    """
    Single dataset evaluation.
    """
    from sklearn.model_selection import train_test_split

    model = load_embedder(model_id)

    data = compute_scores(model, model_id, ds_path, args.method, prompt=prompt)
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


def main(model: str, metric: str, calibration: str, datasets_dir: str, outdir: str, single: bool = False, prompt: str = None):
    os.makedirs(outdir, exist_ok=True)
    if not single:
        datasets = compute_all_datasets(model, datasets_dir, args.method, args.paraphrasus_consistent, prompt=prompt)

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
                threshold_results = loo_eval(datasets, met, "threshold", args.held_out_dataset, calibration_sample_size=args.calibration_sample_size)
                results[f"threshold_{met}"] = threshold_results

                # ---------- Classifier calibration ----------
                print(f"\nRunning {met.upper()} eval with CLASSIFIER calibration")
                classifier_results = loo_eval(datasets, met, "classifier", args.held_out_dataset, calibration_sample_size=args.calibration_sample_size)
                results[f"classifier_{met}"] = classifier_results

        if args.paraphrasus_consistent:
            results_path = os.path.join(outdir, f"{model.replace('/', '_')}_comparable_full_results.json")
        else:
            results_path = os.path.join(outdir, f"{model.replace('/', '_')}_{args.method}_full_results.json")

    elif single:
        print(f"Evaluating single dataset: {args.ds_path.split('/')[-1].replace('.json','')}")
        results = single_eval(model, args.ds_path, metric, calibration, prompt)
        ds_name = os.path.basename(args.ds_path).replace(".json", "")
        results_path = os.path.join(outdir,
                                    f"{model.replace('/', '_')}_{ds_name}_{metric}_{calibration if calibration else ''}"
                                    f"_{args.method if calibration == 'classifier' else ''}_results.json")
    
    else:
        if calibration is None and metric == "auc":
            print("Computing overall AUC...")
            all_labels = np.concatenate([data["labels"] for data in datasets.values()])
            all_scores = np.concatenate([data["scores"] for data in datasets.values()])
            auc = roc_auc_score(all_labels, all_scores)
            results = {"overall_auc": auc}
        else:
            results = loo_eval(datasets, metric, calibration, args.held_out_dataset, calibration_sample_size=args.calibration_sample_size)

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
    parser.add_argument("--method", choices=["elementwise_diff", "multiplication", "sum", "signed_diff", "concatenation"], default="elementwise_diff")
    parser.add_argument("--single_dataset", help="Evaluate on a single dataset", action="store_true")
    parser.add_argument("--ds_path", help="Path to the single dataset JSON file")
    parser.add_argument("--held_out_dataset", help="Specific dataset to hold out in leave-one-out evaluation (optional)")
    parser.add_argument("--prompt", help="Custom prompt for models that support it", default=None)
    parser.add_argument("--calibration_sample_size", type=int, default=500,
                        help="Number of samples per dataset to use for calibration (default: 500)")
    args = parser.parse_args()

    if args.metric in ["error", "f1"] and args.calibration is None and not args.full:
        raise ValueError("Metric 'error' or 'f1' requires a calibration method ('--calibration') or '--full' flag.")
    
    if args.single_dataset and not args.ds_path:
        raise ValueError("Please provide the path to the single dataset using '--ds_path'.")
    
    if args.held_out_dataset and args.single_dataset:
        raise ValueError("Cannot specify both --held_out_dataset and --single_dataset.")
    
    if args.prompt and args.model not in ["intfloat/multilingual-e5-large-instruct", "Qwen/Qwen3-Embedding-0.6B", "intfloat/multilingual-e5-large"]:
        raise Warning("The specified model may not support custom prompts. Proceeding without prompt.")
    
    os.makedirs(args.outdir, exist_ok=True)

    main(args.model, args.metric, args.calibration, args.datasets_dir, args.outdir, args.single_dataset, args.prompt)