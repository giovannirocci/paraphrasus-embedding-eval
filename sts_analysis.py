import os
import argparse
import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from embedding import load_embedder


def compute_similarity(model_id: str, dataset:str, out_path: str, max_samples: int = 500, threshold: float = None):
    ds = datasets.load_dataset(dataset, split="test")
    
    # Sample a subset if the dataset is too large
    n_pairs = len(ds)
    if n_pairs > max_samples // 2:
        import random
        random.seed(42)
        indices = random.sample(range(n_pairs), max_samples // 2)
        ds = ds.select(indices)
        print(f"Sampled {len(ds)} pairs from {n_pairs} total pairs")
    
    model = load_embedder(model_id)

    sentences = list(ds["sentence1"])
    sentences.extend(list(ds["sentence2"]))

    cache_dir = os.path.join("_embedding_cache", model_id.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(dataset)}.npz")

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings for dataset {dataset} from {cache_path}")
        data = np.load(cache_path)
        emb1, emb2 = data["emb1"], data["emb2"]
        # If we sampled, we need to select the corresponding embeddings
        if n_pairs > max_samples // 2:
            emb1 = emb1[indices]
            emb2 = emb2[indices]
        embeddings = np.vstack([emb1, emb2])
    else:
        embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
        np.savez(cache_path, emb1=embeddings[:len(ds)], emb2=embeddings[len(ds):])

    scores = model.similarity(embeddings, embeddings)

    if threshold:
        scores = np.where(scores >= threshold, 1, 0)

    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)

    cbar_kws = {} if not threshold else {"ticks": [0, 1], "boundaries": [0, 0.5, 1]}
    cmap = "viridis_r" if not threshold else ListedColormap(["#fde725", "#440154"])

    plt.rcParams.update({'font.size': 24})

    plt.figure(figsize=(15, 12))
    sns.heatmap(scores, mask=mask, cmap=cmap, xticklabels=False, yticklabels=False, square=True, cbar_kws=cbar_kws)
    
    if threshold:
        plt.title(f"Predictions for {dataset.split('/')[-1].upper()} using {model_id.split('/')[-1]} (Thresholded at {threshold})", pad=20, fontsize=22)
    else:
        plt.title(f"Similarity Matrix for {dataset.split('/')[-1].upper()} using {model_id.split('/')[-1]} (Sampled)", pad=20, fontsize=24)
    
    plt.xlabel("Sentences in Dataset", fontsize=24)
    plt.ylabel("Sentences in Dataset", fontsize=24)
    plt.tight_layout()  
    plt.savefig(out_path, format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='intfloat/multilingual-e5-large-instruct')
    parser.add_argument('--dataset', type=str, default='sentence-transformers/stsb')
    parser.add_argument('--out_path', type=str, default='plots/sts_similarity.pdf')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--threshold', type=float, help="Similarity threshold for considering pairs as paraphrases", required=False)
    args = parser.parse_args()

    compute_similarity(args.model_id, args.dataset, args.out_path, args.max_samples, args.threshold)
