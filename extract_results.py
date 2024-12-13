import json, os, sys
from typing import Optional

import pandas as pd
from collections import defaultdict
from logger import mylog
from benchmarking import dataset_key_to_base_fname, get_bench_path

logger = mylog.get_logger()



def calc(bench_id: str, paths: dict[str, list[str]], method_names_to_prefixes: dict[str, str], expected_prediction: Optional[bool] = False, dataset_to_method_stats= None, is_clf=False):
    bench_path = get_bench_path(bench_id)
    if dataset_to_method_stats is None:
        dataset_to_method_stats = defaultdict(lambda: defaultdict(lambda: {'wrong': 0, 'total': 0}))
    for dataset_key, dataset_paths in paths.items():
        # logger.info(f"Calculating results for {dataset_key}..")
        for file_key in dataset_paths:
            file_path = os.path.join(bench_path, dataset_key_to_base_fname[file_key]+".json")

            with open(file_path, 'r') as f:
                data = json.load(f)
            for sample_id, prediction in data.items():
                for method_name, method_prefix in method_names_to_prefixes.items():
                    for possible_prediction_key, possible_prediction_value in prediction.items():
                        if possible_prediction_key.startswith(method_prefix):
                            if dataset_key == 'STS':
                                score = prediction['score']
                                if score < 0 or score >= 3:
                                    # print(f"ignored score {score} for {dataset_key}")
                                    continue
                            if dataset_key == 'SICK':
                                score = prediction['score']
                                if score < 1 or score >= 3:
                                    # print(f"ignored score {score} for {dataset_key}")
                                    continue
                            # print(expected_prediction)
                            if is_clf:
                                if dataset_key == 'STS-H':
                                    if 'label' not in prediction:
                                        # unlabelled, not part of the dataset
                                        continue
                                    expected_prediction = prediction['label']
                                else:
                                    expected_prediction = (prediction['label'] == 1)
                            dataset_to_method_stats[dataset_key][method_name]['total'] += 1
                            if possible_prediction_value != expected_prediction:
                                dataset_to_method_stats[dataset_key][method_name]['wrong'] += 1
                            elif possible_prediction_value == expected_prediction:
                                pass
                            else:
                                logger.error(
                                    f"Unexpected prediction value {possible_prediction_value} for method {method_name}!")
                                raise Exception(
                                    f"Unexpected prediction value {possible_prediction_value} for method {method_name}!")


    data = {}


    # Populate the data dictionary
    for dataset_key, method_stat_dict in dataset_to_method_stats.items():
        for method_name, stats in method_stat_dict.items():
            wrong = stats["wrong"]
            total = stats["total"]
            error_rate = (wrong / total) * 100



            if dataset_key not in data:
                data[dataset_key] = {}
            if method_name not in data[dataset_key]:
                data[dataset_key][method_name] = {}

            data[dataset_key][method_name] = f"{error_rate:.2f}%"

    df = pd.DataFrame(data)
    logger.info("\n" + df.to_string())


    return data


if __name__ == '__main__':
    min_dataset_group_keys = {
        "SNLI": [
            "stanfordnlp_snli_pre_hyp",
            "stanfordnlp_snli_hyp_pre"
        ],
    "ANLI": [
        "fb_anli_pre_hyp",
        "fb_anli_hyp_pre"
    ],
        "XNLI": [
            "fb_xnli_pre_hyp",
            "fb_xnli_hyp_pre"
        ],
    "STS": ["stsbenchmark"],


    "SICK": ["sickr_sts"]
    }
    max_dataset_group_keys = {
        "TRUE": ["simple_amr"],
        "SIMP": [
            "onestop_all"
        ]

    }

    clf_dataset_group_keys = {
        "PAWSX": [
            "pawsx_test"
        ],
        "STS-H": ["stsbenchmark"],
        "MRPC": ["ms_mrpc"]
    }

    methods = {
        "XLM-RoBERTa-EN-ORIG": "XLM-RoBERTa-EN-ORIG",

        "LLama3 zero-shot P1": "LLama3 zero-shot (Paraph)",
        "LLama3 zero-shot P2": "LLama3 zero-shot (Sem Equiv)",
        "LLama3 zero-shot P3": "LLama3 zero-shot (Ex. Same Content)",
        "LLama3 ICL_4 P1": "LLama3 ICL_4 (Paraph)",
        "LLama3 ICL_4 P2": "LLama3 ICL_4 (Sem Equiv)",
        "LLama3 ICL_4 P3": "LLama3 ICL_4 (Ex. Same Content)",
    }

    if len(sys.argv) == 1:
        print("Hi")
    elif len(sys.argv) == 2:
        print(f"Hi {sys.argv[1]}")
    else:
        print("Invalid usage")

    if len(sys.argv) == 1:
        bench_id = "paper"
    elif len(sys.argv) == 2:
        bench_id = sys.argv[1]
    else:
        print("Please only specify the bench identifier.")
        exit(1)

    min = calc(bench_id, min_dataset_group_keys, methods, expected_prediction=False)
    max = calc(bench_id, max_dataset_group_keys, methods, expected_prediction=True)
    clf = calc(bench_id, clf_dataset_group_keys, methods, expected_prediction=None, is_clf=True)

    results = {
        'CLF': clf,
        'MIN': min,
        'MAX': max
    }
    # collect totals for each method on each kind (kind is min, max, clf)
    kind_totals_by_method = {}
    for kind, dict in results.items():
        kind_totals_by_method[kind] = {}
        for dataset, methods_dict in dict.items():
            for method_name, rate in methods_dict.items():
                if method_name not in kind_totals_by_method[kind]:
                    kind_totals_by_method[kind][method_name] = []
                rate = float(rate.replace("%", ""))
                kind_totals_by_method[kind][method_name].append(rate)

    # calculate average of each method
    for kind, totals_dict in kind_totals_by_method.items():
        for method_name, totals in totals_dict.items():
            if 'total' not in results[kind]:
                results[kind]['total'] = {}
            sum = 0
            for i in totals:
                sum += i
            avg = sum / len(totals)
            results[kind]['total'][method_name] = f"{avg:.2f}%"

    bench_path = get_bench_path(bench_id)
    result_path = os.path.join(bench_path, "results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results for {bench_id} written to {result_path}")
