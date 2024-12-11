import json, os
from typing import Optional

import pandas as pd
from collections import defaultdict
from logger import mylog
from benchmarking import dataset_key_to_base_fname, get_bench_path

logger = mylog.get_logger()


# min

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

        # Calculate average error rates
    error_rates = {}

    for dataset_key, method_stat_dict in dataset_to_method_stats.items():
        # logger.info(f"Errors for {dataset_key}..")
        for method_name, stats in method_stat_dict.items():
            wrong = stats['wrong']
            total = stats['total']
            error_rate = ((wrong / total) * 100)
            error_rates[method_name] = error_rate
            # logger.info(f"Method: {method_name} Error rate: {error_rate:.2f}%")

    # Create a dictionary to hold data for DataFrame
    data = {"Method": []}
    datasets = list(dataset_to_method_stats.keys())
    for dataset in datasets:
        data[dataset] = []

    # Populate the data dictionary
    for dataset_key, method_stat_dict in dataset_to_method_stats.items():
        for method_name, stats in method_stat_dict.items():
            wrong = stats["wrong"]
            total = stats["total"]
            error_rate = (wrong / total) * 100
            if method_name not in data["Method"]:
                data["Method"].append(method_name)
            data[dataset_key].append(f"{error_rate:.2f}%")

    # Ensure all methods have entries for each dataset
    max_methods = len(data["Method"])
    for dataset in datasets:
        while len(data[dataset]) < max_methods:
            data[dataset].append("N/A")

    # Create and display the DataFrame
    df = pd.DataFrame(data)
    df.set_index("Method", inplace=True)

    logger.info("\n" + df.to_string())



    return error_rates


if __name__ == '__main__':
    min_dataset_group_keys = {
    "ANLI": [
        "fb_anli_pre_hyp",
        "fb_anli_hyp_pre"
    ],
    "STS": ["stsbenchmark"],

    "XNLI":[
            "fb_xnli_pre_hyp",
            "fb_xnli_hyp_pre"
        ],
    "SNLI": [
            "stanfordnlp_snli_pre_hyp",
            "stanfordnlp_snli_hyp_pre"
        ],
    "SICK": ["sickr_sts"]
    }
    max_dataset_group_keys = {
        "SIMP": [
            "onestop_all"
        ],
        "TRUE": ["simple_amr"]
    }

    clf_dataset_group_keys = {
        "PAWSX": [
            "pawsx_test"
        ],
        "STS-H": ["stsbenchmark"],
        "MRPC": ["ms_mrpc"]
    }

    # methods = {
    #     "llama_3_8b_ins_q4_k_m":"llama_3_8b_ins_q4_k_m",
    #     "llama_3_3_70b_ins_q8": "llama_3_3_70b_ins_q8"
    # }
    methods = {
        "XLM-RoBERTa-EN-ORIG": "XLM-RoBERTa-EN-ORIG",

        "LLama3 zero-shot P1": "LLama3 zero-shot (Paraph)",
        "LLama3 ICL_4 P1": "LLama3 ICL_4 (Paraph)",
        "LLama3 zero-shot P2": "LLama3 zero-shot (Sem Equiv)",
        "LLama3 ICL_4 P2": "LLama3 ICL_4 (Sem Equiv)",
        "LLama3 zero-shot P3": "LLama3 zero-shot (Ex. Same Content)",
        "LLama3 ICL_4 P3": "LLama3 ICL_4 (Ex. Same Content)",
    }
    # bench_id = "initial_alt"
    bench_id = "paper"
    min = calc(bench_id, min_dataset_group_keys, methods, expected_prediction=False)
    # logger.info("MAX:")
    max = calc(bench_id, max_dataset_group_keys, methods, expected_prediction=True)
    clf = calc(bench_id, clf_dataset_group_keys, methods, expected_prediction=None, is_clf=True)