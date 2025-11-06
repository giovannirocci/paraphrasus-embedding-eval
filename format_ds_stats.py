from pylatex import Document, Tabular, MultiColumn, NoEscape, Package, Command
import re
import numpy as np
import argparse
import os

from embedding import load_dataset


def load_data(ds_dir):
    data_stats = {}
    datasets = [
        os.path.join(ds_dir, d)
        for d in os.listdir(ds_dir)
        if d.endswith(".json")
    ]

    path_to_name = {
        "paws-x-test": "PAWS-X",
        "ms-mrpc": "MRPC",
        "stsbenchmark-test-sts": "STS-H",
        "sickr-sts": "SICK-STS",
        "amr_true_paraphrases": "TRUE",
        "onestop_parallel_all_pairs": "SIMP",
        "tapaco_paraphrases": "TAPACO",
        "stannlp-snli-hyp-pre": "SNLI (hyp-pre)",
        "stannlp-snli-pre-hyp": "SNLI (pre-hyp)",
        "fb-anli-pre-hyp": "ANLI (pre-hyp)",
        "fb-anli-hyp-pre": "ANLI (hyp-pre)",
        "fb-xnli-pre-hyp": "XNLI (pre-hyp)",
        "fb-xnli-hyp-pre": "XNLI (hyp-pre)"
    }

    for ds_path in datasets:
        ds_name = os.path.basename(ds_path).replace(".json", "")
        name = path_to_name[ds_name] if ds_name in path_to_name else ds_name

        if args.unbalanced:
            pairs, labels, goal = load_dataset(ds_path, multi_eval=False)
        else:
            pairs, labels, goal = load_dataset(ds_path, multi_eval=True)
            
        num_pairs = len(pairs)
        num_positive = np.sum(labels)
        pos_rel = num_positive / num_pairs if num_pairs > 0 else 0
        num_negative = num_pairs - num_positive
        neg_rel = num_negative / num_pairs if num_pairs > 0 else 0
        data_stats[name] = {
            "num_pairs": num_pairs,
            "num_positive": num_positive,
            "pos_rel": pos_rel,
            "num_negative": num_negative,
            "neg_rel": neg_rel,
            "goal": goal
        }
    return data_stats


def generate_latex_table(data_stats, output_filepath):
    # Categorize datasets
    classify = ["PAWS-X", "MRPC", "STS-H"]
    minimize = ["SNLI (hyp-pre)", "SNLI (pre-hyp)", "ANLI (hyp-pre)", "ANLI (pre-hyp)", "XNLI (hyp-pre)", "XNLI (pre-hyp)", "SICK-STS"]
    maximize = ["TRUE", "SIMP", "TAPACO"]

    groups = {
        "Classify!": classify,
        "Minimize!": minimize,
        "Maximize!": maximize
    }

    doc = Document()
    doc.packages.append(Package('booktabs'))

    with doc.create(Tabular('r r r r r r')) as table:
        table.append(NoEscape(r'\toprule'))
        table.add_row((
            NoEscape(r'\textbf{Dataset}'),
            NoEscape(r'\textbf{# Pairs}'),
            MultiColumn(2, align='c',data=NoEscape(r'\textbf{Paraphrase}')),
            MultiColumn(2, align='c',data=NoEscape(r'\textbf{¬Paraphrase}')),
        ))
        table.add_row(('', '', 'abs.', 'rel.', 'abs.', 'rel.'))
        table.append(NoEscape(r'\midrule'))

        total_pairs = total_pos = total_neg = 0

        for group_name, group_datasets in groups.items():
            # Group header
            table.add_row((MultiColumn(6, align='c', data=NoEscape(r'\textbf{' + group_name + '}')),))
            table.append(NoEscape(r'\\[-1ex]'))

            group_pairs = group_pos = group_neg = 0

            for ds in group_datasets:
                if ds not in data_stats:
                    continue
                st = data_stats[ds]
                table.add_row((
                    ds,
                    f"{st['num_pairs']:,}",
                    f"{int(st['num_positive']):,}",
                    f"{int(st['pos_rel']*100):.1f}%",
                    f"{int(st['num_negative']):,}",
                    f"{int(st['neg_rel']*100):.1f}%",
                ))
                group_pairs += st['num_pairs']
                group_pos += st['num_positive']
                group_neg += st['num_negative']

            # Group totals
            pos_rel = group_pos / group_pairs if group_pairs else 0
            neg_rel = group_neg / group_pairs if group_pairs else 0
            table.append(NoEscape(r'\midrule'))
            table.add_row((
                NoEscape(r'\textbf{Group}'),
                f"{group_pairs:,}",
                f"{group_pos:,}",
                f"{int(pos_rel*100):.1f}%",
                f"{group_neg:,}",
                f"{int(neg_rel*100):.1f}%"
            ))
            table.append(NoEscape(r'\midrule'))

            total_pairs += group_pairs
            total_pos += group_pos
            total_neg += group_neg

        # Total row
        pos_rel = total_pos / total_pairs if total_pairs else 0
        neg_rel = total_neg / total_pairs if total_pairs else 0
        table.add_row((
            NoEscape(r'\textbf{Total}'),
            f"{total_pairs:,}",
            f"{total_pos:,}",
            f"{int(pos_rel*100):.1f}%",
            f"{total_neg:,}",
            f"{int(neg_rel*100):.1f}%"
        ))
        table.append(NoEscape(r'\bottomrule'))

    doc.generate_tex(output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table of dataset statistics.")
    parser.add_argument("--ds-dir", type=str, default="datasets_no_results", help="Directory containing dataset JSON files.")
    parser.add_argument("--output-filepath", type=str, required=True, help="Output filepath for the LaTeX table (without .tex extension).")
    parser.add_argument("--unbalanced", action="store_true", help="Indicate if datasets are unbalanced.")
    args = parser.parse_args()

    data_stats = load_data(args.ds_dir)
    generate_latex_table(data_stats, args.output_filepath)