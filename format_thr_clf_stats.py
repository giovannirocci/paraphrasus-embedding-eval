from pylatex import Document, Section, Tabular, NoEscape, Package, MultiColumn
import json
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
                        thr_f1.append({"clfy":data[stat]["overall_classify"], 
                                       "min":data[stat]["overall_minimize"], 
                                       "max":data[stat]["overall_maximize"], 
                                       "mean":data[stat]["overall_mean"]})
                    elif stat == "threshold_error":
                        thr_err.append({"clfy":data[stat]["overall_classify"], 
                                        "min":data[stat]["overall_minimize"], 
                                        "max":data[stat]["overall_maximize"], 
                                        "mean":data[stat]["overall_mean"]})
                    elif stat == "classifier_f1":
                        clf_f1.append({"clfy":data[stat]["overall_classify"], 
                                       "min":data[stat]["overall_minimize"],
                                       "max":data[stat]["overall_maximize"], 
                                       "mean":data[stat]["overall_mean"]})
                    elif stat == "classifier_error":
                        clf_err.append({"clfy":data[stat]["overall_classify"], 
                                        "min":data[stat]["overall_minimize"], 
                                        "max":data[stat]["overall_maximize"], 
                                        "mean":data[stat]["overall_mean"]})
                    else:
                        continue
    
    return {
        "threshold": {"f1_clfy" :np.mean([d["clfy"] for d in thr_f1]),
                        "f1_min" :np.mean([d["min"] for d in thr_f1]),
                        "f1_max" :np.mean([d["max"] for d in thr_f1]),
                        "f1_mean" :np.mean([d["mean"] for d in thr_f1]),
                        "error_clfy" :np.mean([d["clfy"] for d in thr_err]),
                        "error_min" :np.mean([d["min"] for d in thr_err]),
                        "error_max" :np.mean([d["max"] for d in thr_err]),
                        "error_mean" :np.mean([d["mean"] for d in thr_err])
                      },
        "classifier": {"f1_clfy" :np.mean([d["clfy"] for d in clf_f1]),
                        "f1_min" :np.mean([d["min"] for d in clf_f1]),   
                        "f1_max" :np.mean([d["max"] for d in clf_f1]),   
                        "f1_mean" :np.mean([d["mean"] for d in clf_f1]),
                        "error_clfy" :np.mean([d["clfy"] for d in clf_err]),
                        "error_min" :np.mean([d["min"] for d in clf_err]),
                        "error_max" :np.mean([d["max"] for d in clf_err]),
                        "error_mean" :np.mean([d["mean"] for d in clf_err])
                      }
    }


def format_stats_to_latex(stats: dict, output_path: str):
    doc = Document()
    doc.packages.append(Package('booktabs'))

    with doc.create(Section('Classification Statistics')):
        doc.append(NoEscape(r'\centering'))
        with doc.create(Tabular('l|r r r r r r r r')) as table:
            table.append(NoEscape(r'\toprule'))
            table.add_row((
                'Calibration Method',
                MultiColumn(4, align='c', data='F1 Score'),
                MultiColumn(4, align='c', data='Error Rate'),
            ))
            table.add_row((
                '',
                NoEscape(r'$\overline{Clfy}$'),
                NoEscape(r'$\overline{Min}$'),
                NoEscape(r'$\overline{Max}$'),
                NoEscape(r'$\overline{F1}$'),
                NoEscape(r'$\overline{Clfy}$'),
                NoEscape(r'$\overline{Min}$'),
                NoEscape(r'$\overline{Max}$'),
                NoEscape(r'$\overline{Err}$'),
            ))
            table.append(NoEscape(r'\midrule'))

            for calibr_name in stats:
                table.add_row((
                    calibr_name.capitalize(),
                    f"{stats[calibr_name]['f1_clfy']:.2f}",
                    f"{stats[calibr_name]['f1_min']:.2f}",
                    f"{stats[calibr_name]['f1_max']:.2f}",
                    f"{stats[calibr_name]['f1_mean']:.2f}",
                    f"{stats[calibr_name]['error_clfy']:.2f}",
                    f"{stats[calibr_name]['error_min']:.2f}",
                    f"{stats[calibr_name]['error_max']:.2f}",
                    f"{stats[calibr_name]['error_mean']:.2f}",
                ))
            table.append(NoEscape(r'\bottomrule'))

    doc.generate_tex(output_path)
    print(f"LaTeX table saved to {output_path}.tex")


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