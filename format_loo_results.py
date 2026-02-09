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
        model_name = input_filepath.split('/')[-1].split('_')[-1]
    else:
        pattern = r'_(elementwise_diff|multiplication|sum)(_(auc|error|full_results))?\.json$'
        input_filepath = re.sub(pattern, '', input_filepath)
        model_name = input_filepath.split('/')[-1].split('_')[-1]

    mapping = {
        "paraphrase-multilingual-mpnet-base-v2": "Para-SBERT",
        "multilingual-e5-large-instruct": "mE5-instr",
        "multilingual-e5-large": "mE5",
        "bge-m3": "BGE-m3",
        "gte-multilingual-base": "mGTE",
        "jina-embeddings-v3": "Jina-v3",
        "Qwen3-Embedding-0.6B": "Qwen3-Emb",
        "KaLM-embedding-multilingual-mini-instruct-v2.5": "KaLM-Emb"
    }

    if model_name in mapping:
        model_name = mapping[model_name]

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
        if k in data:
            merged[k] = data[k]
        else:
            merged[k] = None

    return merged


def fmt(x):
    return f"{x * 100:.1f}" if x is not None else "-"


def create_results_table(input_dir, output_filepath, clf_only=False, f1=False):
    doc = Document()
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('booktabs'))

    col_spec = ('p{2.3cm}|p{0.46cm}p{0.46cm}p{0.46cm}|p{0.46cm}p{0.46cm}p{0.46cm}p{0.46cm}|p{0.46cm}p{0.46cm}p{0.46cm}||'
                'p{0.46cm}p{0.46cm}p{0.46cm}p{0.46cm}')

    with doc.create(Section('Evaluation Results')):
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
                MultiColumn(3, align='c|', data='Maximize!'),
                MultiColumn(4, align='c', data='Averages')
            ))
            table.append(NoEscape(r'\midrule'))

            table.add_row([
                'Model',
                rotate('PAWS-X'), rotate('MRPC'), rotate('STS-H'),
                rotate('SNLI'), rotate('ANLI'), rotate('XNLI'), rotate('SICK-STS'),
                rotate('TRUE'), rotate('SIMP'), rotate('TAPACO'),
                'Clfy', 'Min', 'Max', NoEscape(r'$\overline{F1}$') if f1 else NoEscape(r'$\overline{Err}$')
            ])
            table.append(NoEscape(r'\midrule'))
            
            thresholds, classifiers = [], []
            bests = {}
            for filename in sorted(os.listdir(input_dir)):

                filepath = os.path.join(input_dir, filename)
                data, model_name = load_results(filepath)

                if f1:
                    thr = clean_prepare(data["threshold_f1"])
                    clf = clean_prepare(data["classifier_f1"])
                else:
                    thr = clean_prepare(data["threshold_error"])
                    clf = clean_prepare(data["classifier_error"])

                for i in thr.keys():
                    if i not in bests:
                        bests[i] = thr[i]
                    else:
                        if thr[i] is not None:
                            if f1:
                                if thr[i] > bests[i] and thr[i] > clf[i]:
                                    bests[i] = thr[i]
                                elif clf[i] > bests[i] and clf[i] > thr[i]:
                                    bests[i] = clf[i]
                            else:
                                if thr[i] < bests[i] and thr[i] < clf[i]:
                                    bests[i] = thr[i]
                                elif clf[i] < bests[i] and clf[i] < thr[i]:
                                    bests[i] = clf[i]

                # Threshold calibration row
                row1 = [
                    model_name,
                    fmt(thr.get("PAWS-X")),
                    fmt(thr.get("MRPC")),
                    fmt(thr.get("STS-H")),
                    fmt(thr.get("SNLI")),
                    fmt(thr.get("ANLI")),
                    fmt(thr.get("XNLI")),
                    fmt(thr.get("SICK-STS")),
                    fmt(thr.get("TRUE")),
                    fmt(thr.get("SIMP")),
                    fmt(thr.get("TAPACO")),
                    fmt(thr.get("overall_classify")),
                    fmt(thr.get("overall_minimize")),
                    fmt(thr.get("overall_maximize")),
                    fmt(thr.get("overall_mean")),
                ]
                # Classifier calibration row
                row2 = [
                    model_name,
                    fmt(clf.get("PAWS-X")),
                    fmt(clf.get("MRPC")),
                    fmt(clf.get("STS-H")),
                    fmt(clf.get("SNLI")),
                    fmt(clf.get("ANLI")),
                    fmt(clf.get("XNLI")),
                    fmt(clf.get("SICK-STS")),
                    fmt(clf.get("TRUE")),
                    fmt(clf.get("SIMP")),
                    fmt(clf.get("TAPACO")),
                    fmt(clf.get("overall_classify")),
                    fmt(clf.get("overall_minimize")),
                    fmt(clf.get("overall_maximize")),
                    fmt(clf.get("overall_mean")),
                ]
                
                thresholds.append(row1)
                classifiers.append(row2)

            order = [
                "PAWS-X",
                "MRPC",
                "STS-H",
                "SNLI",
                "ANLI",
                "XNLI",
                "SICK-STS",
                "TRUE",
                "SIMP",
                "TAPACO",
                "overall_classify",
                "overall_minimize",
                "overall_maximize",
                "overall_mean"
            ]

            bests = {k: bests[k] for k in order}

            def get_best(row):
                new = []
                new.append(row[0])
                for i in range(1, len(row)):
                    if row[i] == fmt(list(bests.values())[i-1]):
                        if i == row.index(row[-1]):
                            new.append(NoEscape(r'\textbf{' + row[i] + '}' + r'$\star$'))
                        else:
                            new.append(NoEscape(r'\textbf{' + row[i] + '}'))
                    else:
                        new.append(row[i])
                return new

            if clf_only:
                for row in classifiers:
                    table.add_row(get_best(row))
            else:
                for row in thresholds:
                    table.add_row(get_best(row))
                table.append(NoEscape(r'\midrule'))
                for row in classifiers:
                    table.add_row(get_best(row))
                    
        table.append(NoEscape(r'\bottomrule'))

    if not os.path.exists(output_filepath.split('/')[0]):
        os.makedirs(output_filepath.split('/')[0])

    doc.generate_tex(output_filepath)
    print(f"LaTeX table saved to {output_filepath}.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX results table from JSONs.")
    parser.add_argument("--input_dir", type=str, default="embedding_benchmarks/balanced", help="Directory containing *_full_results.json files")
    parser.add_argument("--output", type=str, default="tables/loo_results", help="Output filename (without .tex)")
    parser.add_argument("--clf_only", action='store_true', help="Generate only classifier calibration results")
    parser.add_argument("--f1", action='store_true', help="Generate F1 score table instead of error rates")
    args = parser.parse_args()

    create_results_table(args.input_dir, args.output, args.clf_only, args.f1)
