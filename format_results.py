from pylatex import Document, Section, Tabularx, MultiColumn, NoEscape, Package, Command
import json
import re
import numpy as np
import argparse
import os


def load_results(input_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'comparable' in input_filepath:
        pattern = r'_comparable_(auc|error|full_results)\.json$'
        input_filepath = re.sub(pattern, '', input_filepath)
        model_name = input_filepath.split('/')[-1] + " (Paraphrasus Consistent)"
    else:
        pattern = r'_(elementwise_diff|multiplication|sum)_(auc|error|full_results)\.json$'
        input_filepath = re.sub(pattern, '', input_filepath)
        model_name = input_filepath.split('/')[-1]

    return data, model_name


def clean_prepare(data):
    merged = {}

    # pairs to merge
    pairs = [
        ("stannlp-snli-hyp-pre", "stannlp-snli-pre-hyp", "SNLI"),
        ("fb-anli-hyp-pre", "fb-anli-pre-hyp", "ANLI"),
        ("fb-xnli-hyp-pre", "fb-xnli-pre-hyp", "XNLI"),
    ]

    for a, b, name in pairs:
        vals = []
        if a in data: vals.append(data[a])
        if b in data: vals.append(data[b])
        merged[name] = np.mean(vals) if vals else None

    # single datasets
    mapping = {
        "paws-x-test": "PAWS-X",
        "ms-mrpc": "MRPC",
        "stsbenchmark-test-sts": "STS-H",
        "sickr-sts": "SICK-STS",
        "amr_true_paraphrases": "TRUE",
        "onestop_parallel_all_pairs": "SIMP",
        "tapaco_paraphrases": "TAPACO",
    }

    for key, name in mapping.items():
        if key in data:
            merged[name] = data[key]
        else:
            merged[name] = None

    # overall scores
    for k in ["overall_classify", "overall_minimize", "overall_maximize", "overall_mean"]:
        merged[k] = data[k]

    return merged


def create_results_table(input_dir, output_filepath):
    doc = Document()
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('booktabs'))

    col_spec = ('p{2.5cm}|p{0.4cm}p{0.4cm}p{0.4cm}|p{0.4cm}p{0.4cm}p{0.4cm}p{0.4cm}|p{0.4cm}p{0.4cm}p{0.4cm}||'
                'p{0.4cm}p{0.4cm}p{0.4cm}p{0.4cm}')

    with doc.create(Section('Evaluation Results')):
        doc.append(NoEscape(r'\centering'))
        doc.append(NoEscape(r'\small'))

        with doc.create(Tabularx(col_spec, width_argument=NoEscape(r'1.2\textwidth'))) as table:

            def rotate(text):
                """Wraps text in a \rotatebox{90}{...} command."""
                return Command('rotatebox', arguments='90', extra_arguments=text)

            table.append(NoEscape(r'\toprule'))
            table.add_row((
                '',
                MultiColumn(3, align='c|', data='Classify!'),
                MultiColumn(4, align='c|', data='Minimize!'),
                MultiColumn(3, align='c|', data='Maximize!'),
                MultiColumn(4, align='c', data='Averages')
            ))
            table.append(NoEscape(r'\midrule'))
            table.add_row([
                'Model',
                rotate('PAWS-X'), rotate('MRPC'), rotate('STS-H'),
                rotate('SNLI'), rotate('ANLI'), rotate('XNLI'), rotate('SICK-STS'),
                rotate('TRUE'), rotate('SIMP'), rotate('TAPACO'),
                'Clfy', 'Min', 'Max', NoEscape(r'$\overline{Err}$')
            ])
            table.append(NoEscape(r'\midrule'))

            def fmt(x):
                return f"{x * 100:.1f}" if x is not None else "-"

            for filename in sorted(os.listdir(input_dir)):
                if not filename.endswith("_full_results.json"):
                    continue

                filepath = os.path.join(input_dir, filename)
                data, model_name = load_results(filepath)
                th_err = clean_prepare(data["threshold_error"])
                cl_err = clean_prepare(data["classifier_error"])

                # Threshold calibration row
                row1 = [
                    model_name + " (Threshold Calib.)",
                    fmt(th_err.get("PAWS-X")),
                    fmt(th_err.get("MRPC")),
                    fmt(th_err.get("STS-H")),
                    fmt(th_err.get("SNLI")),
                    fmt(th_err.get("ANLI")),
                    fmt(th_err.get("XNLI")),
                    fmt(th_err.get("SICK-STS")),
                    fmt(th_err.get("TRUE")),
                    fmt(th_err.get("SIMP")),
                    fmt(th_err.get("TAPACO")),
                    fmt(th_err.get("overall_classify")),
                    fmt(th_err.get("overall_minimize")),
                    fmt(th_err.get("overall_maximize")),
                    fmt(th_err.get("overall_mean")),
                ]
                # Classifier calibration row
                row2 = [
                    model_name + " (Classifier Calib.)",
                    fmt(cl_err.get("PAWS-X")),
                    fmt(cl_err.get("MRPC")),
                    fmt(cl_err.get("STS-H")),
                    fmt(cl_err.get("SNLI")),
                    fmt(cl_err.get("ANLI")),
                    fmt(cl_err.get("XNLI")),
                    fmt(cl_err.get("SICK-STS")),
                    fmt(cl_err.get("TRUE")),
                    fmt(cl_err.get("SIMP")),
                    fmt(cl_err.get("TAPACO")),
                    fmt(cl_err.get("overall_classify")),
                    fmt(cl_err.get("overall_minimize")),
                    fmt(cl_err.get("overall_maximize")),
                    fmt(cl_err.get("overall_mean")),
                ]
                table.add_row(row1)
                table.add_row(row2)

        table.append(NoEscape(r'\bottomrule'))

    doc.generate_tex(output_filepath)
    print(f"LaTeX table saved to {output_filepath}.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX results table from JSONs.")
    parser.add_argument("--input_dir", type=str, default="embedding_benchmarks/full", help="Directory containing *_full_results.json files")
    parser.add_argument("--output", type=str, default="combined_results_table", help="Output filename (without .tex)")
    args = parser.parse_args()

    create_results_table(args.input_dir, args.output)
