from pylatex import Document, Section, Tabular, NoEscape, Package, MultiColumn
import json
import numpy as np
import argparse
import os

from format_loo_results import fmt


def extract_stats(results_dir):
    """
    Extracts the results from the JSON files in the results directory.

    Args:
        results_dir: path to the directory containing the results JSON files
    Returns:
        dict
    """
    results = {}
    out = {}

    mapping = {
            "paraphrase-multilingual-mpnet-base-v2": "M-MPNET",
            "multilingual-e5-large-instruct": "mE5-instr",
            "multilingual-e5-large": "mE5",
            "bge-m3": "BGE-m3",
            "gte-multilingual-base": "mGTE",
            "jina-embeddings-v3": "Jina-v3",
            "Qwen3-Embedding-0.6B": "Qwen3-Emb",
            "KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-Emb",
            "paraphrase-multilingual-MiniLM-L12-v2": "M-MiniLM",
        }

    lr_variant = os.path.basename(results_dir)

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), "r") as f:
                model_name = file.split("_")[1]
                model_name = mapping.get(model_name, model_name)
                data = json.load(f)
                results[model_name] = {metric: data.get(metric, None) for metric in ["overall_classify", "overall_maximize", "overall_minimize", "overall_mean"]}
    out[lr_variant] = results
    return out


def format_results_table(results, outpath):
    """
    Format the results into a LaTeX table.

    Args:
        results: dict of results
    Returns:
        LaTeX table as a string
    """
    doc = Document(documentclass="standalone")
    doc.packages.append(Package("booktabs"))
    doc.packages.append(Package("multirow"))

    with doc.create(Tabular("llcccc")) as table:
        table.append(NoEscape(r'\toprule'))
        table.add_row(('LR Variant',
                        'Model',
                        NoEscape(r'$\overline{Clfy}$'),
                        NoEscape(r'$\overline{Min}$'),
                        NoEscape(r'$\overline{Max}$'),
                        NoEscape(r'$\overline{Err}$'),
                    ))
        table.append(NoEscape(r'\midrule'))
        for variant, variant_results in results.items():
            table.append(NoEscape(r'\multirow{ 9}{*}{\rotatebox{90}{%s}}' % variant))
            for model, metrics in variant_results.items():
                if metrics is not None:
                    row = ['', model] + [fmt(metrics.get(metric, None)) for metric in ["overall_classify", "overall_minimize", "overall_maximize", "overall_mean"]]
                    table.add_row(row)
            table.append(NoEscape(r'\midrule'))
        table.append(NoEscape(r'\bottomrule'))

    doc.generate_tex(outpath)
    print(f"LaTeX table saved to {outpath}.tex")


def main(results_dir, outpath):
    results = {}
    for dir in os.listdir(results_dir):
        dir_path = os.path.join(results_dir, dir)
        if os.path.isdir(dir_path):
            res = extract_stats(dir_path)
            results.update(res)

    format_results_table(results, os.path.join(outpath, f"lr_comparison"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format LR comparison results into a LaTeX table.")
    parser.add_argument("--results_dir", type=str, help="Directory containing the results JSON files.", default="embedding_benchmarks/clf_features")
    parser.add_argument("--outpath", type=str, help="Output path for the LaTeX table.", default="tables")
    args = parser.parse_args()

    main(args.results_dir, args.outpath)