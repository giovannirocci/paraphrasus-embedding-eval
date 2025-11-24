import datasets
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.metrics import f1_score, accuracy_score

from calibration import threshold_learning, classifier_learning
from evaluation import compute_all_datasets
from embedding import load_embedder, compute_pairwise_similarity


def load_data(ds_name: str):
    """
    Load PAWS-X training split dataset from the datasets library.
    """
    if ds_name == "paws-x":
        dataset = datasets.load_dataset("google-research-datasets/paws-x", "en", split="train")
        df = dataset.to_pandas()
    else:
        raise ValueError(f"Dataset {ds_name} not supported.")
    
    pairs = []
    y = []
    for _, row in df.iterrows():
        pairs.append((row['sentence1'], row['sentence2']))
        y.append(row['label'])

    y = np.asarray(y, dtype=np.int32)
    return pairs, y


def get_embeddings(model, pairs):
    """
    Compute embeddings for sentence pairs.
    """
    sentences1 = [p[0] for p in pairs]
    sentences2 = [p[1] for p in pairs]
    emb1 = model.encode(sentences1, show_progress_bar=True)
    emb2 = model.encode(sentences2, show_progress_bar=True)
    return emb1, emb2


def evaluate(thr, clf, model, ds_dir):
    """
    Evaluate threshold and classifier on all datasets in the directory.
    """
    res = compute_all_datasets(model, ds_dir, clf_method="elementwise_diff", paraphrasus_consistent=True)
    results = {}
    for ds in res:
        scores = np.asarray(res[ds]['scores'])
        y = np.asarray(res[ds]['labels'])
        goal = res[ds]['goal']

        # Threshold evaluation
        y_pred_threshold = (scores > thr).astype(np.int32)
        results[ds] = {}
        results[ds]['threshold_learning'] = {
            "f1_score": f1_score(y, y_pred_threshold),
            "error": 1 - accuracy_score(y, y_pred_threshold),
            "threshold": thr
        }

        # Classifier evaluation
        X_diffs = np.asarray(res[ds]['diffs'])
        y_pred_classifier = clf.predict(X_diffs)
        results[ds]['classifier_learning'] = {
            "f1_score": f1_score(y, y_pred_classifier),
            "error": 1 - accuracy_score(y, y_pred_classifier)
        }

        results[ds]['goal'] = goal
    return results


def calculate_means(results):
    thr_classify, thr_minimize, thr_maximize = [], [], []
    classify, minimize, maximize = [], [], []
    for ds in results:
        goal = results[ds]['goal']
        if goal == "classify":
            thr_classify.append(results[ds]['threshold_learning']['error'])
            classify.append(results[ds]['classifier_learning']['error'])
        elif goal == "minimize":
            thr_minimize.append(results[ds]['threshold_learning']['error'])
            minimize.append(results[ds]['classifier_learning']['error'])
        elif goal == "maximize":
            thr_maximize.append(results[ds]['threshold_learning']['error'])
            maximize.append(results[ds]['classifier_learning']['error'])

    results['mean_classify_threshold'] = np.mean(thr_classify) if thr_classify else None
    results['mean_minimize_threshold'] = np.mean(thr_minimize) if thr_minimize else None
    results['mean_maximize_threshold'] = np.mean(thr_maximize) if thr_maximize else None
    results['mean_classify'] = np.mean(classify) if classify else None
    results['mean_minimize'] = np.mean(minimize) if minimize else None
    results['mean_maximize'] = np.mean(maximize) if maximize else None
    
    results['overall_mean_threshold'] = np.mean([results['mean_classify_threshold'], results['mean_minimize_threshold'], results['mean_maximize_threshold']])
    results['overall_mean'] = np.mean([results['mean_classify'], results['mean_minimize'], results['mean_maximize']])
    return results
    

def main():
    parser = argparse.ArgumentParser(description="PAWS-X Training Ablation Study")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the embedding model")
    parser.add_argument("--ds-dir", type=str, default="datasets_no_results", help="Directory containing datasets for evaluation")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the results JSON file")
    args = parser.parse_args()

    # Load PAWS-X training data
    pairs, y = load_data("paws-x")

    # Load embedding model
    model = load_embedder(args.model_name)

    # Compute embeddings
    emb1, emb2 = get_embeddings(model, pairs)

    # Compute pairwise similarity scores
    scores = compute_pairwise_similarity(model, emb1, emb2)

    # Threshold learning
    threshold, _ = threshold_learning(np.asarray(scores), np.asarray(y))

    # Classifier learning
    X_diffs = np.abs(emb1 - emb2)
    classifier = classifier_learning(np.asarray(X_diffs), np.asarray(y))

    # Evaluate results
    res = evaluate(threshold, classifier, args.model_name, args.ds_dir)

    # Compute mean results
    results = calculate_means(res)

    # Save results to JSON
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
