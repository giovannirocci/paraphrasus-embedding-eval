from pylatex import Document, Section, Tabular, NoEscape, Package
import json
import re
import numpy as np
import argparse
import os


def extract_means(res_dir: str):
    thr_f1 = []
    thr_err = []
    clf_f1 = []
    clf_err = []

    for file in os.listdir(res_dir):
        if file.endswith(".json"):
            with open(os.path.join(res_dir, file), "r") as f:
                data = json.load(f)
                for stat in data:
                    if stat == "threshold_f1":
                        thr_f1.append(data[stat]["overall_mean"])
                    elif stat == "threshold_error":
                        thr_err.append(data[stat]["overall_mean"])
                    elif stat == "classifier_f1":
                        clf_f1.append(data[stat]["overall_mean"])
                    elif stat == "classifier_error":
                        clf_err.append(data[stat]["overall_mean"])
                    else:
                        continue
    
    return {
        "threshold": {"f1" :np.mean(thr_f1), "error": np.mean(thr_err)},
        "classifier": {"f1": np.mean(clf_f1), "error": np.mean(clf_err)},
    }


def format_stats_to_latex(stats: dict, output_path: str):
    doc = Document()
    doc.packages.append(Package('booktabs'))

    with doc.create(Section('Classification Statistics')):
        doc.append(NoEscape(r'\centering'))
        with doc.create(Tabular('l|r r')) as table:
            table.append(NoEscape(r'\toprule'))
            table.add_row((
                'Calibration Method',
                NoEscape(r'$\overline{F1}$'),
                NoEscape(r'$\overline{Err}$'),
            ))
            table.append(NoEscape(r'\midrule'))

            for calibr_name in stats:
                table.add_row((
                    calibr_name.capitalize(),
                    f"{stats[calibr_name]['f1']:.4f}",
                    f"{stats[calibr_name]['error']:.4f}",
                ))
            table.append(NoEscape(r'\bottomrule'))

    doc.generate_tex(output_path)
    print(f"LaTeX report generated at {output_path}.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format calibration statistics into a LaTeX table."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing JSON result files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the LaTeX file (without extension).",
    )

    args = parser.parse_args()

    stats = extract_means(args.results_dir)
    format_stats_to_latex(stats, args.output_path)