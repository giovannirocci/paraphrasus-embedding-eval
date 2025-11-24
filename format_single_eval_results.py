import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_path: str) -> pd.DataFrame:
    """Load nested JSON results into a flat DataFrame."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for ds_name, models in results.items():
        for model_name, metrics in models.items():
            row = {
                "dataset": ds_name,
                "model": model_name,
                **metrics,  # auc_none, f1_threshold, error_threshold, f1_classifier, error_classifier
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add nicer labels for models and datasets."""
    df = df.copy()

    df["dataset_clean"] = df["dataset"].replace({
        "ms-mrpc": "MRPC",
        "paws-x-test": "PAWS-X",
        "stsbenchmark-test-sts": "STS-H",
    })

    df["model_clean"] = df["model"].replace({
        "Alibaba-NLP/gte-multilingual-base": "mGTE",
        "BAAI/bge-m3": "BGE-m3",
        "intfloat/multilingual-e5-large-instruct": "multilingual-E5",
        "jinaai/jina-embeddings-v3": "Jina-v3",
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-mini-v2.5",
        "paraphrase-multilingual-mpnet-base-v2": "paraphrase-SBERT",
        "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Emb-0.6B",
    })

    return df


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, cbar_label: str,
                 out_path: Path):
    """Plot a heatmap for the specified metric."""
    pivot_df = df.pivot(
        index="model_clean",
        columns="dataset_clean",
        values=value_col,
    )

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def make_latex_table(df: pd.DataFrame, out_path: Path):
    """Create a LaTeX table with all metrics."""
    df_tex = df.rename(columns={
        "dataset": "Dataset",
        "model": "Model",
        "auc_none": "AUC",
        "f1_threshold": "F1$_{thr}$",
        "error_threshold": "Err$_{thr}$",
        "f1_classifier": "F1$_{clf}$",
        "error_classifier": "Err$_{clf}$",
    })

    cols_order = [
        "Dataset", "Model", "AUC",
        "F1$_{thr}$", "Err$_{thr}$",
        "F1$_{clf}$", "Err$_{clf}$",
    ]
    df_tex = df_tex[cols_order]

    latex_str = df_tex.to_latex(
        index=False,
        float_format="%.3f",
        escape=False,  # keep math in column names
    )

    out_path.write_text(latex_str, encoding="utf-8")
    print(f"Saved LaTeX table to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="embedding_benchmarks/single_dataset/results.json",
        help="Path to the results JSON file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load and prepare data
    df = load_results(args.results_path)
    df = clean_labels(df)

    # 2) Two heatmaps: threshold error and classifier error
    plot_heatmap(
        df,
        value_col="error_threshold",
        title="Threshold calibration error across datasets",
        cbar_label="Error (threshold)",
        out_path=outdir / "heatmap_error_threshold.png",
    )

    plot_heatmap(
        df,
        value_col="error_classifier",
        title="Classifier calibration error across datasets",
        cbar_label="Error (classifier)",
        out_path=outdir / "heatmap_error_classifier.png",
    )

    # 3) LaTeX table with all metrics
    make_latex_table(df, Path("tables/single_dataset_results.tex"))


if __name__ == "__main__":
    main()
