import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os


# -------------------------------
# Embedding utility
# -------------------------------

def load_embedder(model_name: str):
    """Load an Embedding model"""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
    return model


def compute_pairwise_similarity(model, emb1, emb2):
    try:
        return model.similarity_pairwise(emb1, emb2)
    except AttributeError:
        import torch
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

def compute_scores(model: SentenceTransformer, model_name:str, dataset_path: str, classifier_method: str):
    """
    Compute similarity scores and element-wise difference for all pairs in the dataset using the specified model.
    Args:
        model: the embedding model
        model_name: name of the embedding model (for caching purposes)
        dataset_path: path to the dataset JSON file
        classifier_method: method to compute differences ("elementwise_diff", "multiplication", "sum")
    Returns:
        dict with keys: scores, diffs, labels, goal
    """
    pairs, labels, goal = load_dataset(dataset_path)
    texts1, texts2 = zip(*pairs)

    cache_dir = os.path.join("_embedding_cache", model_name.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(dataset_path)}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        emb1, emb2 = data["emb1"], data["emb2"]
    else:
        emb1, emb2 = model.encode(texts1), model.encode(texts2)
        np.savez(cache_path, emb1=emb1, emb2=emb2)

    if classifier_method == "elementwise_diff":
        diffs = np.abs(emb1 - emb2)
    elif classifier_method == "multiplication":
        diffs = emb1 * emb2
    elif classifier_method == "sum":
        diffs = emb1 + emb2
    else:
        raise ValueError(f"Unknown classifier method: {classifier_method}")

    return {
        "scores": compute_pairwise_similarity(model, emb1, emb2),
        "diffs": diffs,
        "labels": labels,
        "goal": goal
    }
