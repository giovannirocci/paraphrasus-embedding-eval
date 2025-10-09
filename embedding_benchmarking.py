import importlib
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score

from logger import mylog  # already in the repo


log = mylog.get_logger()


# ---------- Dataset loading ----------

def load_dataset(ds_path: str) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Load different datasets from JSON files.
    """
    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classify =["paws-x-test",
               "ms-mrpc",
               "stsbenchmark-test-sts"
               ]

    minimize = ["stannlp-snli-hyp-pre",
                "stannlp-snli-pre-hyp",
                "fb-anli-pre-hyp",
                "fb-anli-hyp-pre",
                "fb-xnli-pre-hyp",
                "fb-xnli-hyp-pre",
                "sickr-sts"
                ]

    maximize = [
        "amr_true_paraphrases",
        "onestop_parallel_all_pairs",
    ]

    try:
        pairs = [(x["sentence1"], x["sentence2"]) for x in data.values()]
    except KeyError:
        raise ValueError(f"Cannot find 'sentence1' and 'sentence2' keys in dataset: {ds_path}, "
                         f"dataset sentence pairs must be under these keys.")

    if "stsbenchmark-test-sts" in ds_path:
        # Special case: we take from STS benchmark only labeled pairs
        labels = []
        pairs = []
        for x in data.values():
            if "label" in x:
                labels.append(x["label"])
                pairs.append((x["sentence1"], x["sentence2"]))
        y = np.asarray([1 if l == True else 0 for l in labels], dtype=np.int32)
    else:
        if any(key in ds_path for key in classify):
            # Classification datasets: "label" is boolean
            y = np.asarray([1 if x["label"] == True else 0 for x in data.values()], dtype=np.int32)
        elif any(key in ds_path for key in minimize):
            # Minimization datasets: ground truth will always be 0
            y = np.zeros((len(pairs),), dtype=np.int32)
        elif any(key in ds_path for key in maximize):
            # Maximization datasets: ground truth will always be 1
            y = np.ones((len(pairs),), dtype=np.int32)
        else:
            raise ValueError(f"Cannot infer dataset type from path: {ds_path}")

    if not len(pairs) == len(y):
        raise ValueError(f"Number of pairs and labels do not match in dataset: {ds_path}")

    return pairs, y

# ---------- Threshold selection ----------

def choose_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    objective: str = "f1",  # "f1" | "error"
) -> float:
    """
    Grid-search a cosine threshold in [-1,1] on training set.
    """
    grid = np.linspace(-1.0, 1.0, 2001)  # 0.001 resolution
    best_thr = 0.0
    best_val = -1.0 if objective == "f1" else 1e9

    # Sort once to speed iteration (optional)
    # But brute-force is fine at this resolution.
    for thr in grid:
        y_pred = (scores >= thr).astype(np.int32)
        if objective == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
            if val > best_val:
                best_val = val
                best_thr = thr
        else:
            err = (y_pred != y_true).mean()
            if err < best_val:
                best_val = err
                best_thr = thr
    return best_thr

# ---------- Leave-one-out over datasets ----------

def loo_evaluate(
    datasets: Dict[str, Tuple[List[Tuple[str, str]], np.ndarray]],
    method_mod: str,
    method_fun: str,
    method_kwargs: Dict,
    is_dummy: bool,
    benches_dir: str,
    bench_id: str,
    objective: str = "f1",
) -> Dict:
    """
    For each held-out dataset D:
      1) Compute cosine scores for every dataset via the method.
      2) Fit a single threshold on concatenated scores from all datasets except D.
      3) Apply that threshold to D and compute metrics.
    Saves a results.json in benches/<bench_id>/.
    """
    os.makedirs(os.path.join(benches_dir, bench_id), exist_ok=True)

    # Load method dynamically
    mod = importlib.import_module(method_mod)
    score_fn = getattr(mod, method_fun)

    # Precompute all scores once (with caching inside method if requested)
    ds_names = list(datasets.keys())
    scores_by_ds: Dict[str, np.ndarray] = {}
    y_by_ds: Dict[str, np.ndarray] = {}

    log.info(f"Scoring all datasets with {method_mod}.{method_fun} ...")

    for ds in ds_names:
        pairs, y = datasets[ds]
        if is_dummy:
            log.info(f"Skipping actual scoring on {ds} (dummy method) ...")
            scores = np.random.uniform(-1.0, 1.0, size=(len(pairs),)).astype(np.float32)
        else:
            scores = score_fn(pairs, **method_kwargs)  # np.ndarray (N,)
        scores_by_ds[ds] = scores.astype(np.float32, copy=False)
        y_by_ds[ds] = y

    # LOO
    results = {}
    all_test_rows = []  # to also compute macro/weighted summaries if you like

    log.info(f"Running leave-one-out thresholding (objective={objective}) ...")
    for test_ds in ds_names:
        # concatenate train scores/labels
        train_scores = []
        train_y = []
        for ds in ds_names:
            if ds == test_ds:
                continue
            train_scores.append(scores_by_ds[ds])
            train_y.append(y_by_ds[ds])
        train_scores = np.concatenate(train_scores, axis=0)
        train_y = np.concatenate(train_y, axis=0)

        thr = choose_threshold(train_scores, train_y, objective=objective)

        # Evaluate on held-out
        test_scores = scores_by_ds[test_ds]
        test_y = y_by_ds[test_ds]
        y_pred = (test_scores >= thr).astype(np.int32)

        err = float((y_pred != test_y).mean())
        acc = 1.0 - err
        f1 = float(f1_score(test_y, y_pred, zero_division=0))

        results[test_ds] = {
            "scores": test_scores.tolist(),
            "threshold": float(thr),
            "error": err,
            "accuracy": acc,
            "f1": f1,
            "n": int(test_y.shape[0]),
        }
        all_test_rows.append((err, acc, f1))

    # Macro averages
    if all_test_rows:
        arr = np.asarray(all_test_rows, dtype=np.float32)
        results["_macro"] = {
            "error": float(arr[:, 0].mean()),
            "accuracy": float(arr[:, 1].mean()),
            "f1": float(arr[:, 2].mean()),
        }

    # Save results
    out_path = os.path.join(benches_dir, bench_id, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"Saved LOO results to {out_path}")
    return results

# ---------- CLI-like entry via config ----------

def run_from_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bench_id = cfg["bench_id"]
    benches_dir = cfg.get("benches_dir", "embedding_evaluations")
    os.makedirs(benches_dir, exist_ok=True)
    objective = cfg.get("objective", "f1")  # or "error"

    # Load datasets (paths)
    # Expect: "datasets": {"DSNAME": "path/to/dataset.json", ...}
    ds_paths: Dict[str, str] = cfg["datasets"]
    datasets = {name: load_dataset(path) for name, path in ds_paths.items()}

    # Methods: you may evaluate multiple, but here we do one at a time for clarity
    for m in cfg["methods"]:
        name = m["name"]
        module = m["module"]
        function = m["function"]
        dummy = m.get("dummy", False)
        method_kwargs = m.get("kwargs", {})

        # Allow separate result directory per method
        sub_bench_id = f"{bench_id}_{name}"
        loo_evaluate(
            datasets=datasets,
            method_mod=module,
            method_fun=function,
            method_kwargs=method_kwargs,
            is_dummy=dummy,
            benches_dir=benches_dir,
            bench_id=sub_bench_id,
            objective=objective,
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 embedding_evaluation.py <config.json>")
        raise SystemExit(2)
    run_from_config(sys.argv[1])
