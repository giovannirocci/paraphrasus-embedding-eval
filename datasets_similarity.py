import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse


from embedding import load_dataset, load_embedder


def compute_dataset_similarity(model_id: str, datasets_dir: str, out_path: str):
    """
    Compute similarity between different datasets based on embedding distances.
    """
    model = load_embedder(model_id)
    dataset_names = []
    dataset_embeddings = {}

    print("Loading datasets and computing embeddings...")
    for ds_file in os.listdir(datasets_dir):
        if ds_file.endswith(".json"):
            ds_path = os.path.join(datasets_dir, ds_file)
            ds_name = ds_file.replace(".json", "")
            pairs, _, _ = load_dataset(ds_path)
            sentences = [st for pair in pairs for st in pair]

            # Compute embeddings for all sentences in the dataset
            cache_dir = os.path.join("_embedding_cache", model_id.replace("/", "_"))
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{os.path.basename(ds_name)}.npz")

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                emb1, emb2 = data["emb1"], data["emb2"]
                embeddings = np.vstack([emb1, emb2])
            else:
                embeddings = model.encode(sentences)


            dataset_names.append(ds_name)
            avg_embedding = np.mean(embeddings, axis=0)
            dataset_embeddings[ds_name] = avg_embedding

    # Compute similarity matrix using cosine similarity
    num_datasets = len(dataset_names)
    matrix = np.zeros((num_datasets, num_datasets))
    for i in range(num_datasets):
        for j in range(num_datasets):
            emb_i = dataset_embeddings[dataset_names[i]]
            emb_j = dataset_embeddings[dataset_names[j]]
            cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            matrix[i, j] = cos_sim

    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=dataset_names, yticklabels=dataset_names, annot=True, fmt=".2f", cmap="Purples", mask=mask)
    plt.title(f"Dataset Similarity Matrix using {model_id}")
    plt.tight_layout()

    plt.savefig(out_path)
    print(f"Dataset similarity heatmap saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and visualize dataset similarity based on embeddings.")
    parser.add_argument("--model", type=str, required=True, help="Embedding model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--datasets_dir", type=str, default="datasets_no_results", help="Directory containing dataset JSON files")
    parser.add_argument("--out_path", type=str, default="plots/dataset_similarity_heatmap.png", help="Output path for the heatmap image")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    compute_dataset_similarity(args.model, args.datasets_dir, args.out_path)
