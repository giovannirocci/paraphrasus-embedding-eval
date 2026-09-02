from pylatex import Document, Section, Tabularx, MultiColumn, NoEscape, Package, Command
import json
import numpy as np
import argparse

from format_loo_results import fmt

def prepare(input_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = {"M-MPNET":{"thr":{}, "lr-abs":{}},
        "mE5-instr":{"thr":{}, "lr-abs":{}},
        "mE5":{"thr":{}, "lr-abs":{}},
        "BGE-m3":{"thr":{}, "lr-abs":{}},
        "mGTE":{"thr":{}, "lr-abs":{}},
        "Jina-v3":{"thr":{}, "lr-abs":{}},
        "Qwen3-Emb":{"thr":{}, "lr-abs":{}},
        "KaLM-Emb":{"thr":{}, "lr-abs":{}},
        "M-MiniLM":{"thr":{}, "lr-abs":{}}
    }

    for group in data:
        if group == "Averages":
            continue

        for dataset in data[group]:
            for model in data[group][dataset]:
                model_name = model.split("/")[0]
                calibration = model.split("/")[1]

                if "P1" in model_name or "P2" in model_name or "P3" in model_name:
                    continue
                
                if calibration == "thr":
                    models[model_name]["thr"][dataset] = float(data[group][dataset][model].strip('%'))
                elif calibration == "lr-abs":
                    models[model_name]["lr-abs"][dataset] = float(data[group][dataset][model].strip('%'))
                else:
                    continue
    
    for model in data["Averages"]:
        model_name = model.split("/")[0]
        calibration = model.split("/")[1]

        if "P1" in model_name or "P2" in model_name or "P3" in model_name:
            continue

        if calibration == "thr":
            models[model_name]["thr"]["clfy"] = float(data["Averages"][model]["Classify!"].strip('%'))
            models[model_name]["thr"]["min"] = float(data["Averages"][model]["Minimize!"].strip('%'))
            models[model_name]["thr"]["max"] = float(data["Averages"][model]["Maximize!"].strip('%'))
            models[model_name]["thr"]["overall_err"] = float(data["Averages"][model]["Overall Average"].strip('%'))
        elif calibration == "lr-abs":
            models[model_name]["lr-abs"]["clfy"] = float(data["Averages"][model]["Classify!"].strip('%'))
            models[model_name]["lr-abs"]["min"] = float(data["Averages"][model]["Minimize!"].strip('%'))
            models[model_name]["lr-abs"]["max"] = float(data["Averages"][model]["Maximize!"].strip('%'))
            models[model_name]["lr-abs"]["overall_err"] = float(data["Averages"][model]["Overall Average"].strip('%'))
        else:
            continue

    thr_averages = {"clfy": float(fmt(np.mean([models[m]["thr"]["clfy"] for m in models if "thr" in models[m]]))) / 100,
                    "min": float(fmt(np.mean([models[m]["thr"]["min"] for m in models if "thr" in models[m]]))) / 100,
                    "max": float(fmt(np.mean([models[m]["thr"]["max"] for m in models if "thr" in models[m]]))) / 100,
                    "overall_err": float(fmt(np.mean([models[m]["thr"]["overall_err"] for m in models if "thr" in models[m]]))) / 100}

    lr_averages = {"clfy": float(fmt(np.mean([models[m]["lr-abs"]["clfy"] for m in models if "lr-abs" in models[m]]))) / 100,
                    "min": float(fmt(np.mean([models[m]["lr-abs"]["min"] for m in models if "lr-abs" in models[m]]))) / 100,
                    "max": float(fmt(np.mean([models[m]["lr-abs"]["max"] for m in models if "lr-abs" in models[m]]))) / 100,
                    "overall_err": float(fmt(np.mean([models[m]["lr-abs"]["overall_err"] for m in models if "lr-abs" in models[m]]))) / 100}

    print("Threshold Average Errors:", thr_averages)
    print("Logistic Regression Average Errors:", lr_averages)

    return models


def generate_latex_table(models, output_filepath):
    doc = Document()
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('booktabs'))

    col_spec = ('p{2.3cm}|p{0.46cm}p{0.46cm}p{0.46cm}|p{0.46cm}p{0.46cm}p{0.46cm}p{0.46cm}p{0.46cm}|p{0.46cm}p{0.46cm}||'
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
                MultiColumn(5, align='c|', data='Minimize!'),
                MultiColumn(2, align='c|', data='Maximize!'),
                MultiColumn(4, align='c', data='Averages')
            ))
            table.append(NoEscape(r'\midrule'))

            table.add_row([
                'Model',
                rotate('PAWS-X'), rotate('MRPC'), rotate('STS-H'),
                rotate('SNLI'), rotate('ANLI'), rotate('XNLI'), rotate('STS'), rotate('SICK'),
                rotate('TRUE'), rotate('SIMP'),
                'Clfy', 'Min', 'Max', NoEscape(r'$\overline{Err}$')
            ])
            table.append(NoEscape(r'\midrule'))

            ordered_models = ['M-MiniLM', 'M-MPNET', 'mGTE', 'KaLM-Emb', 'mE5', 'mE5-instr', 'BGE-m3', 'Jina-v3', 'Qwen3-Emb']
            for model_name in ordered_models:
                if model_name in models:
                    model_data = models[model_name]
                    row = [model_name]
                    for dataset in ['PAWSX', 'MRPC', 'STS-H', 'SNLI', 'ANLI', 'XNLI', 'STS', 'SICK', 'TRUE', 'SIMP']:
                        row.append(model_data.get('thr', {}).get(dataset, None))
                    row.append(model_data.get('thr', {}).get('clfy', None))
                    row.append(model_data.get('thr', {}).get('min', None))
                    row.append(model_data.get('thr', {}).get('max', None))
                    row.append(model_data.get('thr', {}).get('overall_err', None))
                    table.add_row(row)

            table.append(NoEscape(r'\midrule'))

            for model_name in ordered_models:
                if model_name in models:
                    model_data = models[model_name]
                    row = [model_name]
                    for dataset in ['PAWSX', 'MRPC', 'STS-H', 'SNLI', 'ANLI', 'XNLI', 'STS', 'SICK', 'TRUE', 'SIMP']:
                        row.append(model_data.get('lr-abs', {}).get(dataset, None))
                    row.append(model_data.get('lr-abs', {}).get('clfy', None))
                    row.append(model_data.get('lr-abs', {}).get('min', None))
                    row.append(model_data.get('lr-abs', {}).get('max', None))
                    row.append(model_data.get('lr-abs', {}).get('overall_err', None))
                    table.add_row(row)

            table.append(NoEscape(r'\bottomrule'))

        doc.generate_tex(output_filepath)
        print(f"LaTeX table saved to {output_filepath}.tex")


def main(input_filepath, output_filepath):
    models = prepare(input_filepath)
    generate_latex_table(models, output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table from JSON results.")
    parser.add_argument("-i","--input_filepath", type=str, help="Path to the input JSON file.")
    parser.add_argument("-o","--output_filepath", type=str, help="Path to the output LaTeX file (without .tex extension).")
    args = parser.parse_args()

    main(args.input_filepath, args.output_filepath)