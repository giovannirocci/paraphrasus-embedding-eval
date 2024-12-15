from openai import OpenAI
from benchmarking import bench
from logger import mylog
import pandas as pd
import sys, json, importlib

logger = mylog.get_logger()





# Load configuration file
def load_config(config_path):
    with open(config_path, "r") as config_file:
        return json.load(config_file)

# Locate and fetch the method
def get_method(config_entry):
    module_name = config_entry["module"]
    function_name = config_entry["function"]

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Get the function from the module
    func = getattr(module, function_name)

    if not callable(func):
        raise ValueError(f"{function_name} in {module_name} is not callable")

    return func

if __name__ == '__main__':
    if len(sys.argv) == 1:
        config_path = "paper_config.json"
    elif len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        print("Please specify a config path!")
        exit(1)

    config = load_config(config_path)
    bench_id = config["bench_id"]

    m = config["methods"]
    methods = {}
    for method in m:
        n = method["name"]

        func = get_method(method)

        methods[n] = func

    bench(
        methods=methods,
    bench_id=bench_id,
    batch_size=1,
    )