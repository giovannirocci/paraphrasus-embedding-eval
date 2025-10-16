import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import json
import os
from tqdm import tqdm

from calibration import threshold_learning, classifier_learning
from embedding import compute_scores, load_embedder


def compute_all_datasets(model_id: str, calibration: str, datasets_dir: str, outdir: str):
    """
    Compute scores for all datasets in the specified directory.
    """
    os.makedirs(outdir, exist_ok=True)
    model = load_embedder(model_id)

    results = {}
    print("Computing similarity scores...")
    for ds_file in tqdm(os.listdir(datasets_dir), total=len(os.listdir(datasets_dir))):
        if ds_file.endswith(".json"):
            ds_path = os.path.join(datasets_dir, ds_file)
            ds_name = ds_file.replace(".json", "")
            scores, labels, goal = compute_scores(model, model_id, ds_path, calibration)
            results[ds_name] = {
                "scores": scores,
                "labels": labels,
                "goal": goal
            }
    return results


def evaluate_loo_threshold(datasets, metric: str):
    """
    Leave-One-Out evaluation with threshold calibration.
    """
    results = {}
    print("Performing Leave-One-Out evaluation with threshold calibration...")
    for held_out in datasets:
        train_scores = []
        train_labels = []
        for ds_name, data in datasets.items():
            if ds_name != held_out:
                train_scores.extend(data["scores"])
                train_labels.extend(data["labels"])
        best_thr, _ = threshold_learning(np.array(train_scores), np.array(train_labels))

        print(f"Evaluating on held-out dataset: {held_out}")
        held_out_data = datasets[held_out]
        scores = held_out_data["scores"]
        if hasattr(scores, "detach"):  # torch.Tensor
            scores = scores.detach().cpu().numpy()
        else:
            scores = np.asarray(scores, dtype=np.float32)  # normal numpy array

        preds = (scores > best_thr).astype(np.int32)

        if metric == "f1":
            f1 = f1_score(held_out_data["labels"], preds, zero_division=1)
            results[held_out] = f1
        elif metric == "error":
            acc = accuracy_score(held_out_data["labels"], preds)
            error = 1 - acc
            results[held_out] = error
        else:
            raise ValueError(f"Unknown metric {metric}")
    return results


def evaluate_loo_classifier(datasets, metric: str):
    """
    Leave-One-Out evaluation with classifier calibration.
    """
    results = {}
    print("Performing Leave-One-Out evaluation with classifier calibration...")
    for held_out in datasets:
        train_scores = []
        train_labels = []
        for ds_name, data in datasets.items():
            if ds_name != held_out:
                train_scores.extend(data["scores"])
                train_labels.extend(data["labels"])
        clf = classifier_learning(np.vstack(train_scores), np.array(train_labels))

        print(f"Evaluating on held-out dataset: {held_out}")
        held_out_data = datasets[held_out]
        preds = clf.predict(np.array(held_out_data["scores"]))

        if metric == "f1":
            f1 = f1_score(held_out_data["labels"], preds, zero_division=1)
            results[held_out] = f1
        elif metric == "error":
            acc = accuracy_score(held_out_data["labels"], preds)
            error = 1 - acc
            results[held_out] = error

        else:
            raise ValueError(f"Unknown metric {metric}")
    return results


def main(model: str, metric: str, calibration: str, datasets_dir: str, outdir: str):
    datasets = compute_all_datasets(model, calibration, datasets_dir, outdir)

    results = {}
    if args.full:
        for met in ["auc", "f1", "error"]:
            if met == "auc":
                all_labels = np.concatenate([data["labels"] for data in datasets.values()])
                all_scores = np.concatenate([data["scores"] for data in datasets.values()])
                auc = roc_auc_score(all_labels, all_scores)
                results[f"overall_{met}"] = auc
            else:
                for cal in ["threshold", "classifier"]:
                    if cal == "threshold":
                        res = evaluate_loo_threshold(datasets, met)
                    elif cal == "classifier":
                        res = evaluate_loo_classifier(datasets, met)
                    results[f"{cal}_{met}"] = res
        results_path = os.path.join(outdir, f"{model.replace('/', '_')}_full_results.json")
    else:
        if calibration == "threshold":
            results = evaluate_loo_threshold(datasets, metric)
        elif calibration == "classifier":
            results = evaluate_loo_classifier(datasets, metric)
        elif calibration is None:
            all_labels = np.concatenate([data["labels"] for data in datasets.values()])
            all_scores = np.concatenate([data["scores"] for data in datasets.values()])
            auc = roc_auc_score(all_labels, all_scores)
            results = {"overall_auc": auc}
        else:
            raise ValueError(f"Unknown calibration method: {calibration}")
        results_path = os.path.join(outdir, f"{model.replace('/', '_')}_{metric}_{calibration if calibration else ''}_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", choices=["auc", "error", "f1"])
    parser.add_argument("--calibration", choices=["threshold", "classifier"])
    parser.add_argument("--datasets_dir", default="datasets_no_results")
    parser.add_argument("--outdir", default="embedding_benchmarks")
    parser.add_argument("--full", action="store_true", help="Evaluate on all metrics and all calibration methods")
    args = parser.parse_args()

    if args.metric in ["error", "f1"] and args.calibration is None and not args.full:
        raise ValueError("Metric 'error' or 'f1' requires a calibration method ('--calibration') or '--full' flag.")

    main(args.model, args.metric, args.calibration, args.datasets_dir, args.outdir)
