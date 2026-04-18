import os
import argparse
import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time

from embedding import load_embedder


def compute_similarity(model_id: str, dataset:str, out_path: str, max_samples: int = 500, threshold: float = None, color_1: str = None, color_2: str = None):
    start_time = time.time()
    
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
    n_sentences = len(sentences)
    print(f"Total sentences to process: {n_sentences}")

    cache_dir = os.path.join("_embedding_cache", model_id.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(dataset)}.npz")

    embed_start = time.time()
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
    embed_time = time.time() - embed_start
    print(f"Embedding computation time: {embed_time:.2f} seconds")

    sim_start = time.time()
    scores = model.similarity(embeddings, embeddings)
    sim_time = time.time() - sim_start
    print(f"Similarity computation time: {sim_time:.2f} seconds")

    if threshold:
        scores = np.where(scores >= threshold, 1, 0)

    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)

    cmap = "viridis_r" if not threshold else ListedColormap([color_1, color_2])

    plt.rcParams.update({'font.size': 24})

    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(scores, mask=mask, cmap=cmap, xticklabels=False, yticklabels=False, square=True, cbar=False)
    
    #ax.xaxis.set_label_position('top')
    #ax.yaxis.set_label_position('right')
    
    if threshold:
        legend_elements = [
            mpatches.Patch(facecolor=color_1, label="Not Paraphrase"),
            mpatches.Patch(facecolor=color_2, label="Paraphrase")
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=24)
    
    if threshold:
        plt.title(f"Predictions for {dataset.split('/')[-1].upper()} using {model_id.split('/')[-1]} (Thresholded at {threshold})", pad=50, fontsize=22)
    else:
        plt.title(f"Similarity Matrix for {dataset.split('/')[-1].upper()} using {model_id.split('/')[-1]} (Sampled)", pad=20, fontsize=24)
    
    ax.set_xlabel("Sentences", fontsize=22)
    ax.set_ylabel("Sentences", fontsize=22)
    plt.tight_layout()  
    plt.savefig(out_path, format="pdf")
    
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='intfloat/multilingual-e5-large-instruct')
    parser.add_argument('--dataset', type=str, default='sentence-transformers/stsb')
    parser.add_argument('--out_path', type=str, default='plots/sts_similarity.png')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--threshold', type=float, help="Similarity threshold for considering pairs as paraphrases", required=False)
    parser.add_argument("--color_1", type=str, choices=["#fcba03", "#460457", "#0F172A", "#FB7185", "#1A1B26", "#2AC3DE"], default="#FFF77E", help="First color for thresholded heatmap")
    parser.add_argument("--color_2", type=str, choices=["#fcba03", "#460457", "#0F172A", "#FB7185", "#1A1B26", "#2AC3DE"], default="#460457", help="Second color for thresholded heatmap")
    args = parser.parse_args()

    compute_similarity(args.model_id, args.dataset, args.out_path, args.max_samples, args.threshold, color_1=args.color_1, color_2=args.color_2)
