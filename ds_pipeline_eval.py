import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from embedding import load_embedder, compute_scores
from calibration import threshold_learning, classifier_learning


def load_models(model_dir: str):
    """
    Load model identifiers from a directory.
    """
    return [f.replace("_", "/") for f in os.listdir(model_dir) if not f.startswith(".")]


def evaluate(datasets_dir: str, models_dir: str, out_dir: str, method: str = "elementwise_diff"):
    """
    Optimized evaluation loop:
    - loads each embedding model once
    - caches embeddings on disk
    - avoids repeated calls to evaluation.main()
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect datasets and models
    datasets = [
        os.path.join(datasets_dir, d)
        for d in os.listdir(datasets_dir)
        if d.endswith(".json")
    ]
    models = load_models(models_dir)

    results = {}

    for model_name in tqdm(models, desc="Models"):
        print(f"\n=== Evaluating model: {model_name} ===")
        model = load_embedder(model_name)

        for ds_path in tqdm(datasets, desc="Datasets"):
            ds_name = os.path.basename(ds_path).replace(".json", "")
            print(f"\n→ Dataset: {ds_name}")
            
            classify =["paws-x-test",
               "ms-mrpc",
               "stsbenchmark-test-sts"
               ]
            if ds_name not in classify:
                print(f"Skipping dataset {ds_name} as it is not suitable for single evaluation (contains only 1 class).")
                continue
            else:
                print(f"Processing dataset {ds_name}.")

                # Compute or load embeddings
                data = compute_scores(model, model_name, ds_path, method, multi_eval=False)
                y = np.asarray(data["labels"], dtype=np.int32)
                scores = np.asarray(data["scores"], dtype=np.float32)
                X_diffs = np.asarray(data["diffs"], dtype=np.float32)

                results.setdefault(ds_name, {})[model_name] = {}

                # ---------- AUC ----------
                auc = roc_auc_score(y, scores)
                results[ds_name][model_name]["auc_none"] = float(auc)

                # ---------- Threshold calibration ----------
                X_thr_train, X_thr_test, y_thr_train, y_thr_test = train_test_split(
                    scores, y, test_size=0.2, random_state=42)
                
                thr, _ = threshold_learning(X_thr_train, y_thr_train)
                preds_thr = (X_thr_test > thr).astype(np.int32)
                
                results[ds_name][model_name]["f1_threshold"] = float(
                    f1_score(y_thr_test, preds_thr, zero_division=1)
                )
                results[ds_name][model_name]["error_threshold"] = float(
                    1 - accuracy_score(y_thr_test, preds_thr)
                )

                # ---------- Classifier calibration ----------
                X_clfs_train, X_clfs_test, y_clfs_train, y_clfs_test = train_test_split(
                    X_diffs, y, test_size=0.2, random_state=42)
                
                clf = classifier_learning(X_clfs_train, y_clfs_train)
                preds_clf = clf.predict(X_clfs_test)
                
                results[ds_name][model_name]["f1_classifier"] = float(
                    f1_score(y_clfs_test, preds_clf, zero_division=1)
                )
                results[ds_name][model_name]["error_classifier"] = float(
                    1 - accuracy_score(y_clfs_test, preds_clf)
                )

        # free GPU memory for the next model
        del model

    # --------- SAVE RESULTS ---------
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    datasets_directory = "datasets_no_results"
    models_directory = "_embedding_cache"
    output_directory = "embedding_benchmarks/single_dataset"

    evaluate(
        datasets_dir=datasets_directory,
        models_dir=models_directory,
        out_dir=output_directory,
        method="elementwise_diff",
    )
