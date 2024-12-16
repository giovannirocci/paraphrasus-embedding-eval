import json, os, sys, importlib
from typing import Optional


import pandas as pd
from collections import defaultdict
from logger import mylog
from benchmarking import dataset_key_to_base_fname, get_bench_path

logger = mylog.get_logger()

PRINT_LATEX = False

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

            data[dataset_key][method_name] = f"{error_rate:.1f}%"

    # df = pd.DataFrame(data)
    # logger.info("\n" + df.to_string())


    return data




# Load configuration file
def load_config(config_path):
    with open(config_path, "r") as config_file:
        return json.load(config_file)



if __name__ == '__main__':
    clf_dataset_group_keys = {
        "PAWSX": [
            "pawsx_test"
        ],
        "STS-H": ["stsbenchmark"],
        "MRPC": ["ms_mrpc"]
    }


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





    if len(sys.argv) == 1:
        config_path = "paper_config.json"
    elif len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        print("Please specify a config path!")
        exit(1)

    config = load_config(config_path)
    bench_id = config["bench_id"]
    if "prefixes" in config:
        prefixes = config["prefixes"]
    else:
        m = config["methods"]
        prefixes = {}
        for method in m:
            n = method["name"]
            prefixes[n] = n

    min = calc(bench_id, min_dataset_group_keys, method_names_to_prefixes=prefixes, expected_prediction=False)
    max = calc(bench_id, max_dataset_group_keys, method_names_to_prefixes=prefixes, expected_prediction=True)
    clf = calc(bench_id, clf_dataset_group_keys, method_names_to_prefixes=prefixes, expected_prediction=None, is_clf=True)

    results = {
        'Classify!': clf,
        'Minimize!': min,
        'Maximize!': max
    }
    kind_totals_by_method = {}
    for kind, dict in results.items():
        kind_totals_by_method[kind] = {}
        for dataset, methods_dict in dict.items():
            for method_name, rate in methods_dict.items():
                if method_name not in kind_totals_by_method[kind]:
                    kind_totals_by_method[kind][method_name] = []
                rate = float(rate.replace("%", ""))
                kind_totals_by_method[kind][method_name].append(rate)


    avg_by_method_by_kind = {}



    # calculate average of each method by kind (kind is min, max, clf)
    # and grand total for each method
    grand_totals_by_method = {}
    for kind, totals_dict in kind_totals_by_method.items():
        for method_name, totals in totals_dict.items():
            sum=0
            for i in totals:
                sum += i
            avg = sum/len(totals)
            if method_name not in grand_totals_by_method:
                grand_totals_by_method[method_name] = []
            grand_totals_by_method[method_name].append(avg)

            if method_name not in avg_by_method_by_kind:
                avg_by_method_by_kind[method_name] = {}
            avg_by_method_by_kind[method_name][kind] = f"{avg:.1f}%"

    b1 = {}
    for method_name, totals_dict in avg_by_method_by_kind.items():
        b1[method_name] = 0
        for kind, avg in totals_dict.items():
            b1[method_name] += float(avg.replace("%", ""))
        b1[method_name] = b1[method_name] / 3

    for method_name, a in b1.items():
        avg_by_method_by_kind[method_name]["Overall Average"] = f"{a:.1f}%"



    results["Averages"] = avg_by_method_by_kind

    final_keys_ordering = [
        "Classify!",
        "Minimize!",
        "Maximize!",
        "Overall Average"
    ]


    for method_name in avg_by_method_by_kind.keys():
        s = method_name
        s+="\n"
        for method in clf_dataset_group_keys.keys():
            # print(method)
            rate = results["Classify!"][method][method_name]
            rate = rate.replace("%", "")
            s+=f"& {rate} "
        for method in min_dataset_group_keys.keys():
            # print(method)
            rate = results["Minimize!"][method][method_name]
            rate = rate.replace("%", "")
            s += f"& {rate} "
        for method in max_dataset_group_keys.keys():
            # print(method)
            rate = results["Maximize!"][method][method_name]
            rate = rate.replace("%", "")
            s += f"& {rate} "
        for method in final_keys_ordering:
            # print(method)
            rate = avg_by_method_by_kind[method_name][method]
            rate = rate.replace("%", "")
            s += f"& {rate} "

        if PRINT_LATEX:
            print(s+"\n")




    bench_path = get_bench_path(bench_id)
    result_path = os.path.join(bench_path, "results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results for {bench_id} written to {result_path}")
