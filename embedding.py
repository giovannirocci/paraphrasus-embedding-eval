import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer
import json
import os


# -------------------------------
# Embedding utility
# -------------------------------

def load_embedder(model_name: str):
    """Load an Embedding model"""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model


def compute_pairwise_similarity(model, emb1, emb2):
    try:
        return model.similarity_pairwise(emb1, emb2)
    except AttributeError:
        emb1_t = torch.as_tensor(emb1)
        emb2_t = torch.as_tensor(emb2)
        return torch.nn.functional.cosine_similarity(emb1_t, emb2_t, dim=1).cpu().numpy()

# --------------------------------
# Load dataset utility
# --------------------------------

def load_dataset(ds_path: str):
    """
    Load different datasets from JSON files.
    """
    print(f"Loading dataset: {ds_path.split('/')[-1].rstrip('.json')}")
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
        "tapaco_paraphrases"
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
        goal = "classify"
    else:
        if any(key in ds_path for key in classify):
            # Classification datasets: "label" is boolean
            y = np.asarray([1 if x["label"] == True else 0 for x in data.values()], dtype=np.int32)
            goal = "classify"
        elif any(key in ds_path for key in minimize):
            # Minimization datasets: ground truth will always be 0
            y = np.zeros((len(pairs),), dtype=np.int32)
            goal = "minimize"
        elif any(key in ds_path for key in maximize):
            # Maximization datasets: ground truth will always be 1
            y = np.ones((len(pairs),), dtype=np.int32)
            goal = "maximize"
        else:
            raise ValueError(f"Cannot infer dataset type from path: {ds_path}")

    if not len(pairs) == len(y):
        raise ValueError(f"Number of pairs and labels do not match in dataset: {ds_path}")

    return pairs, y, goal


# -------------------------------
# Score embedding pairs
# -------------------------------

def compute_scores(model: SentenceTransformer, dataset_path: str, calibration: str | None = None):
    """
    Compute similarity scores for all pairs in the dataset using the specified model.
    If calibration is specified, use the corresponding method to compute scores.
    Args:
        model: the embedding model
        dataset_path: path to the dataset JSON file
        calibration: "threshold", "classifier", or None
    """
    pairs, labels, goal = load_dataset(dataset_path)
    texts1, texts2 = zip(*pairs)
    emb1, emb2 = model.encode(texts1), model.encode(texts2)

    if calibration == "threshold" or calibration is None:
        # Learn threshold through cosine similarity / use scores for AUC
        scores = compute_pairwise_similarity(model, emb1, emb2)
        return scores, labels, goal
    elif calibration == "classifier":
        # Learn classifier through element-wise subtraction
        X = np.abs(emb1 - emb2)  # element-wise absolute difference
        return X, labels, goal
    else:
        raise ValueError("Calibration method must be 'threshold', 'classifier', or None.")
