from pylatex import Document, Tabularx, MultiColumn, NoEscape, Package, Command
import json
import numpy as np
import argparse
import os
 

def clean_prepare(data):
    merged = {}

    # pairs to merge
    pairs = [
        ("stannlp-snli-hyp-pre", "stannlp-snli-pre-hyp", "SNLI"),
        ("fb-anli-hyp-pre", "fb-anli-pre-hyp", "ANLI"),
        ("fb-xnli-hyp-pre", "fb-xnli-pre-hyp", "XNLI"),
    ]

    for a, b, name in pairs:
        clf_err = []
        thr_err = []
        if a in data: clf_err.append(data[a]["classifier_learning"]["error"]); thr_err.append(data[a]["threshold_learning"]["error"])
        if b in data: clf_err.append(data[b]["classifier_learning"]["error"]); thr_err.append(data[b]["threshold_learning"]["error"])
        merged[name] = {
            "classifier_learning_error": np.mean(clf_err) if clf_err else None,
            "threshold_learning_error": np.mean(thr_err) if thr_err else None
        }

    # single datasets
    mapping = {
        "paws-x-test": "PAWS-X",
        "ms-mrpc": "MRPC",
        "stsbenchmark-test-sts": "STS-H",
        "sickr-sts": "SICK-STS",
        "amr_true_paraphrases": "TRUE",
        "onestop_parallel_all_pairs": "SIMP"
    }

    for key, name in mapping.items():
        if key in data:
            merged[name] = {
                "classifier_learning_error": data[key]["classifier_learning"]["error"],
                "threshold_learning_error": data[key]["threshold_learning"]["error"]
            }
        else:
            merged[name] = None

    # overall scores
    for k in ["mean_classify_threshold", "mean_minimize_threshold", "mean_maximize_threshold",
              "mean_classify", "mean_minimize", "mean_maximize", "overall_mean_threshold", "overall_mean"]:
        merged[k] = data[k]

    return merged


def create_results_table(input_file, output_filepath):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cleaned = clean_prepare(data)

    doc = Document(documentclass='article', document_options='12pt')
    doc.packages.append(Package('geometry', options=['a4paper', 'margin=1in']))
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('booktabs'))

    col_spec = ('p{2.3cm}|p{0.46cm}p{0.46cm}p{0.6cm}|p{0.46cm}p{0.46cm}p{0.46cm}p{0.6cm}|p{0.46cm}p{0.6cm}||'
                'p{0.46cm}p{0.46cm}p{0.46cm}p{0.46cm}')
    
    doc.append(NoEscape(r'\centering'))
    doc.append(NoEscape(r'\small'))

    with doc.create(Tabularx(col_spec, width_argument=NoEscape(r'\textwidth'))) as table:
        
        def rotate(text):
            """Wraps text in a \rotatebox{90}{...} command."""
            return Command('rotatebox', arguments='90', extra_arguments=text)

        table.append(NoEscape(r'\toprule'))
        table.add_row((
            '',
            MultiColumn(3, align='c|', data='Classify!'),
            MultiColumn(4, align='c|', data='Minimize!'),
            MultiColumn(2, align='c|', data='Maximize!'),
            MultiColumn(4, align='c', data='Averages')
        ))
        table.append(NoEscape(r'\midrule'))
        table.add_row([
            'Model',
            rotate('PAWS-X'), rotate('MRPC'), rotate('STS-H'),
            rotate('SNLI'), rotate('ANLI'), rotate('XNLI'), rotate('SICK-STS'),
            rotate('TRUE'), rotate('SIMP'),
            'Clfy', 'Min', 'Max', NoEscape(r'$\overline{Err}$')
        ])
        table.append(NoEscape(r'\midrule'))

        def fmt(x):
            return f"{x * 100:.1f}" if x is not None else "-"

        # Add data rows
        col_order = [
            "PAWS-X", "MRPC", "STS-H",
            "SNLI", "ANLI", "XNLI", "SICK-STS",
            "TRUE", "SIMP"
        ]

        thr_row = [fmt(cleaned[ds]["threshold_learning_error"]) for ds in col_order if not type(cleaned[ds]) == float]
        clf_row = [fmt(cleaned[ds]["classifier_learning_error"]) for ds in col_order if not type(cleaned[ds]) == float]

        thr_row.extend([
            fmt(cleaned["mean_classify_threshold"]),
            fmt(cleaned["mean_minimize_threshold"]),
            fmt(cleaned["mean_maximize_threshold"]),
            fmt(cleaned["overall_mean_threshold"])
        ])
        clf_row.extend([
            fmt(cleaned["mean_classify"]),
            fmt(cleaned["mean_minimize"]),
            fmt(cleaned["mean_maximize"]),
            fmt(cleaned["overall_mean"])
        ])

        table.add_row(['BGE-m3 (thr.)'] + thr_row)
        table.add_row(['BGE-m3 (clf.)'] + clf_row)
        table.append(NoEscape(r'\bottomrule'))

    doc.generate_tex(output_filepath)
    print(f"LaTeX table saved to {output_filepath}.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format PAWS-X Evaluation Results into LaTeX Table")
    parser.add_argument("--input-file", type=str, default="embedding_benchmarks/paws_train_results.json", help="Path to the input JSON results file")
    parser.add_argument("--output-filepath", type=str, default="tables/paws_eval_results", help="Path to save the output LaTeX file (without .tex extension)")
    args = parser.parse_args()

    create_results_table(args.input_file, args.output_filepath)

            