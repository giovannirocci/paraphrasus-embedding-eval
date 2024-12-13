import os, json, time, shutil, random
import sys
from datetime import datetime

from logger import mylog
from typing import Callable, List, Tuple, Dict, Optional, Any
from progress_bar import print_progress_bar

logger = mylog.get_logger()



SOURCE_DATASET_DIRECTORY = 'datasets_no_results'


def copy_directory_contents(src: str, dst: str, item_list: Optional[List[str]] = None):
    """
        Copies contents from the source directory to the destination directory.

        If `item_list` is specified, only the files listed in `item_list` will be copied.
        If a file already exists in the destination directory, it will *not* be overwritten.

        Parameters
        ----------
        src : str
            Path to the source directory.
        dst : str
            Path to the destination directory.
        item_list : list of str, optional
            if specified, only these files from the src will be copied. Otherwise, all files are copied.

        Notes
        -----
        - This function is idempotent: if the same files are copied multiple times, only new files
          will be added to the destination directory.
        - Only files are copied; directories within the source directory are ignored.
    """
    # Ensure the destination directory exists
    os.makedirs(dst, exist_ok=True)

    # copy the wanted files (if a list is specified,
    # otherwise copy all)
    copied_count = 0
    for item in os.listdir(src):
        if ((item_list is not None and item in item_list)
                or (item_list is None)):
            src_path = os.path.join(src, item)
            dst_path = os.path.join(dst, item)

            # we don't want to replace an existing file.
            # running a bench with the same parameters is idempotent.
            if os.path.exists(dst_path):
                logger.info(f"not copying '{item}' because it already exists in '{dst}'.")
                continue
            # Check if it's a directory or a file
            if not os.path.isdir(src_path):
                # Copy individual files
                shutil.copy2(src_path, dst_path)
                copied_count += 1
    if copied_count > 0:
        logger.info(f"Copied {copied_count} datasets to {dst}.")


def generate_bench_identifier():
    """
    Generates a unique run identifier of the format 'r_XX_YY_ZZ'.
    XX: Day of the month
    YY: Month
    ZZ: Current minute
    """
    now = datetime.now()
    identifier = f"r_{now.day:02d}_{now.month:02d}_{now.minute:02d}"
    return identifier


def get_project_root():
    """
    Retrieves the root directory of the current project.
    Assumes the script is being run from somewhere within the project structure.
    """
    return os.path.abspath(os.getcwd())


def get_bench_path(bench_id: str) -> str:
    """
    Creates a directory for the benchmark under the project root in 'runs/r_XX_YY_ZZ'.
    A bench contains runs of different methods.
    Creates the directory if it doesn't exist.
    """
    # Define directory paths
    project_root = get_project_root()
    benches_directory = os.path.join(project_root, "benches")
    bench_directory = os.path.join(benches_directory, bench_id)

    # Create directories if they don't exist
    os.makedirs(bench_directory, exist_ok=True)

    return bench_directory


# each dataset has an identifier (referred to as 'key')
# this maps that, to the base filename of the dataset, without the extension.
# the datasets exist both in json and in tsv (original code), hence the need for this.
dataset_key_to_base_fname = {
    'stsbenchmark': 'stsbenchmark-test-sts',
    'ms_mrpc': 'ms-mrpc',
    'onestop_all': 'onestop_parallel_all_pairs',
    'simple_amr': 'amr_true_paraphrases',
    'fb_anli_pre_hyp': 'fb-anli-pre-hyp',
    'fb_anli_hyp_pre': 'fb-anli-hyp-pre',
    'sickr_sts': 'sickr-sts',
    'pawsx_test': 'paws-x-test',
    'stanfordnlp_snli_pre_hyp': 'stannlp-snli-pre-hyp',
    'fb_xnli_pre_hyp': 'fb-xnli-pre-hyp',
    'fb_xnli_hyp_pre': 'fb-xnli-hyp-pre',
    'stanfordnlp_snli_hyp_pre': 'stannlp-snli-hyp-pre'
}

def method_check(name, method):
    """
        Makes sure that a prediction method works as expected, giving it multiple retries if errors occur.

        This function ensures the given `method`:
        - given a list of string tuples: 1. doesn't crash 2. returns a list of booleans, and 3. the number of booleans given equal the number of tuples given to it.

        Parameters
        ----------
        name : str
            The name of the method being checked, used for descriptive error messages.
        method : Callable[[List[Tuple[str, str]]], List[bool]]
            The prediction method to validate. It should accept a list of string tuples as input and return a list of booleans.

        Input Samples
        -------------
        The method is tested using the following input samples:
        - [("sample1", "sample2"), ("sample3", "sample4")]

        The function will retry up to 10 times if validation fails due to an exception or incorrect output.

        Raises
        ------
        Exception
            If the method fails validation after all retries, an exception is raised with a descriptive error message detailing the issue.

        Examples
        --------
        >>> def example_method(samples):
        ...     return [True, False]
        >>> method_check("Example Method", example_method)  # Passes without errors

        >>> def invalid_method(samples):
        ...     return ["yes", "no"]
        >>> method_check("Invalid Method", invalid_method)
        Exception: Invalid Method function should return a list of booleans! Returned predictions contain non-boolean values: ['yes', 'no'].

        """
    retries_left = 10
    err = f"{name} function should return a list of booleans! "
    samples = [
            ("sample1", "sample2"),
            ("sample3", "sample4")
        ]
    while retries_left > 0:
        try:
            predictions = method(samples)
        except Exception as e:
            err += "Exception: " + str(e)
            raise Exception(err)
        # are we given a list of booleans as expected?
        if isinstance(predictions, list):
            non_boolean_values = [item for item in predictions if not isinstance(item, bool)]
            if not non_boolean_values:
                # then this batch was correctly processed
                # is it the correct length tho?
                # returned list should be equal to the given prediction samples.
                # otherwise, since we can't correctly associate the prediction result with the original tuple, we have to discard it.
                if len(predictions) != len(samples):
                    err += f"method was given {len(samples)} predictions but {len(predictions)} were returned."
                    if retries_left > 0:
                        retries_left -= 1
                    else:
                        raise Exception(err)
                else:
                    # all is good, let's quit
                    return
            else:
                # Error: the list contains non-boolean values
                err += (
                    f"Returned predictions contain non-boolean values: {non_boolean_values}."
                )
                if retries_left > 0:
                    retries_left -= 1
                else:
                    raise Exception(err)
        else:
            # Error: predictions is not a list
            err += f"Returned predictions is not a list. I was instead given: {predictions!r}."
            if retries_left > 0:
                retries_left -= 1
            else:
                raise Exception(err)


def bench(
        methods: Dict[
                str,
                Callable[[List[Tuple[str, str]]], List[bool]]
                ]
          ,dataset_keys: Optional[list[str]] = None
          , bench_id: str = None, batch_size: int = 64,
          purge_results: bool = False):
    """
        The main benchmarking process.

        This function:
        - Validates the prediction methods using `method_check`, (see above).
        - Runs predictions on all (or the specified subset of) the datasets in paraphrasus.
        - Logs progress and handles errors during predictions, retrying as needed for each batch.
        - Saves results for each method and dataset in the benchmark's dedicated directory (identified by `bench_id`).

        Parameters
        ----------
        methods : dict of str -> Callable[[List[Tuple[str, str]]], List[bool]]
            A dictionary mapping method names to their corresponding prediction functions.
            Each prediction function should accept a list of sentence tuples and return a list of booleans.

        dataset_keys : list of str, optional
            Keys identifying which datasets to benchmark. By default, all datasets are processed.

        bench_id : str, optional
            Identifier for this benchmarking session. If None, a new identifier is generated.
            This function is idempotent, error-reselient and entirely resumable.
            Given the same bench_id, interrupting and running this function will continue where it left off.
            Note: if execution is interrupted during writing of the results, in such a way that the json formatting becomes invalid,
            the next time it tries to resume, it will overwrite the said json file. Which means progress for that specific dataset will be reset.

        batch_size : int, default=64
            Number of samples to process in each batch. This is how many samples are given to each prediction method.

        purge_results : bool, default=False
            If True, removes existing predictions from the dataset before running the benchmark.

        Raises
        ------
        Exception
            If a prediction method fails validation repeatedly, or dataset reading/writing encounters an error.

        Notes
        -----
        - This function creates a unique directory for storing benchmark results (if not already there).

        Example Workflow
        ----------------
        >>> def example_method(batch):
        ...     return [True, False] * (len(batch) // 2)
        >>> methods = {"Example": example_method}
        >>> bench(methods, dataset_keys=["key1", "key2"])

        """

    method_keys = list(methods.keys())
    # FIRST! are the given methods correct?
    for method_key in method_keys:
        method = methods[method_key]
        method_check(method_key, method)
    def read_records():
        try:
            with open(fpath, 'r') as f:
                records: Dict[str, Dict[str, Any]] = json.load(f)
            return records
        except Exception as e:
            # why can this fail? if writing to this file was interrupted, resulting in non-valid JSON formatting.
            # In such case, we have to delete the problematic json, and copy over a fresh one (that doesn't contain any predictions).
            logger.error(f"An error occurred while reading the file: {e}. Will delete it and replace it.")
            # attempt to delete the file
            try:
                os.remove(fpath)
                print(f"File {fpath} deleted successfully. Replacing with a fresh one.")
                base_name = os.path.basename(fpath)
                copy_directory_contents(SOURCE_DATASET_DIRECTORY, bench_dir, [base_name])
                return None
            except Exception as delete_error:
                logger.error(f"An error occurred while deleting the file: {delete_error}. Program will exit.")
                sys.exit(1)
    def save_records():
        with open(fpath, 'w') as f:
            json.dump(records, f, indent=2)

    # create a new directory for the results of this bench.
    # then, copy the all (or only specified) datasets,
    # where the predictions will be written

    # by default copy all datasets
    wanted_filenames = None
    dataset_keys_actual = dataset_key_to_base_fname.keys()
    if dataset_keys is not None:
        wanted_filenames = [f"{dataset_key_to_base_fname[key]}.json" for key in dataset_keys]
        dataset_keys_actual = dataset_keys
    if bench_id is None:
        bench_id = generate_bench_identifier()
    bench_dir = get_bench_path(bench_id)

    copy_directory_contents(src=SOURCE_DATASET_DIRECTORY, dst=bench_dir, item_list=wanted_filenames)

    for dataset_key in dataset_keys_actual:
        # ANSI escape codes for bold blue text
        BOLD_BLUE = "\033[1;34m"
        RESET = "\033[0m"

        logger.info(f"\n\n{BOLD_BLUE}Predicting for {dataset_key}...{RESET}")
        fname = f"{dataset_key_to_base_fname[dataset_key]}.json"
        fpath = os.path.join(bench_dir, fname)

        retries_left = 3
        records=None
        while retries_left > 0 and records is None:
            records = read_records()
        if records is None:
            logger.info(f"Could nor read records for dataset {dataset_key}. Skipping.")
            continue
        if purge_results:
            for method_key in method_keys:
                logger.info(f"Purging results of {method_key}...")
                keys_to_remove = [key for key, value in records.items() if method_key in value.keys()]
                logger.info(f"Removed {len(keys_to_remove)} predictions of {method_key}.")
                for key in keys_to_remove:
                    del records[key][method_key]
            save_records()

        for method_key in method_keys:
            method = methods[method_key]
            # filter the samples that haven't been predicted by this method.
            records_not_predicted = {k: v for k, v in records.items() if method_key not in v.keys()}
            # early exit if this method is done
            if len(records_not_predicted) == 0:
                logger.info(f"No predictions left for method {method_key}.")
                continue

            logger.info(f"{len(records_not_predicted)} predictions to be made using {method_key}...")

            # this is to preserve ordering, making it easier to split batches.
            record_keys = list(records_not_predicted.keys())
            # split the keys into batches.
            batches = []
            current_batch = []
            for r_key in record_keys:
                if len(current_batch) == batch_size:
                    batches.append(current_batch)
                    current_batch = []
                current_batch.append(r_key)
            if len(current_batch) > 0:
                batches.append(current_batch)

            #### timekeeping ####
            batch_durations = []
            processed_count = 0
            #### timekeeping ####

            ### running predictions ###
            total_start_time = time.time()
            for batch_keys in batches:
                # a list of sentence tuples.
                batch: list[tuple[str, str]] = [(
                    str(records_not_predicted[k]["sentence1"]),
                    str(records_not_predicted[k]["sentence2"])
                )
                    for k in batch_keys]
                retries_left=3
                should_retry=True
                predictions=None
                error_msg = ""
                while should_retry and retries_left>0:
                    try:
                        batch_start_time = time.time()
                        predictions = method(batch)
                        batch_end_time = time.time()
                    except Exception as e:
                        error_msg = "Exception: "+str(e)
                        predictions = None
                        retries_left -= 1
                        continue

                    # are we given a list of booleans as expected?
                    if isinstance(predictions, list):
                        non_boolean_values = [item for item in predictions if not isinstance(item, bool)]
                        if not non_boolean_values:
                            # then this batch was correctly processed
                            # is it the correct length tho?
                            # returned list should be equal to the given prediction samples.
                            # otherwise, since we can't correctly associate the prediction result with the original tuple, we have to discard it.
                            if len(predictions) != len(batch_keys):
                                error_msg = f"method was given {len(batch)} predictions but {len(predictions)} were returned."
                                retries_left -= 1
                                predictions = None
                                continue
                            should_retry = False
                        else:
                            # Error: the list contains non-boolean values
                            error_msg = (
                                f"Returned predictions contain non-boolean values: {non_boolean_values}."
                            )
                            predictions = None
                            retries_left -= 1
                    else:
                        # Error: predictions is not a list
                        error_msg = f"Returned predictions is not a list. Value: {predictions!r}."
                        predictions = None
                        retries_left -= 1

                if predictions is None:
                    logger.error(f"Unable to get predictions for method {method_key}: {error_msg}")
                    continue
                processed_count += len(predictions)

                #### SAVING BATCH RESULTS ####
                for i in range(len(batch_keys)):
                    record_key = batch_keys[i]
                    prediction = predictions[i]
                    records[record_key][method_key] = prediction
                save_records()
                #### SAVING BATCH RESULTS ####

                #### TIMEKEEPING ####

                batch_duration = batch_end_time - batch_start_time
                batch_durations.append(batch_duration)
                # Estimate total time taken so far

                elapsed_total = batch_end_time - total_start_time
                # Average units per second so far
                units_per_sec = processed_count / elapsed_total if elapsed_total > 0 else float('inf')
                units_left = len(record_keys) - processed_count
                # def print_progress_bar(name: str, processed_count: int, total_units: int, elapsed_secs: float):
                print_progress_bar(name=method_key, processed_count=processed_count, total_units=len(record_keys),
                                   elapsed_secs=elapsed_total)

                #### TIMEKEEPING ####



def predict_method1(pairs):
    # Example mock implementation:
    # Replace this with your actual prediction logic
    time.sleep(0.01 * len(pairs))
    return [False for _ in pairs]

def predict_method2(pairs):
    # Simulate processing time
    time.sleep(0.005 * len(pairs))
    return [True for _ in pairs]

def predict_method_incomplete(pairs):
    # Simulate processing time
    time.sleep(0.005 * len(pairs))
    return [True for _ in pairs][:-2]

def predict_method_wrongtype_nonlist(pairs):
    # Simulate processing time
    return "some other bs"

def predict_method_wrongtype(pairs):
    # Simulate processing time
    time.sleep(0.005 * len(pairs))
    result =  [True for _ in pairs]
    times = 5
    for i in range(times):
        index_to_replace = random.randint(0, len(result) - 1)  # Choose a random index
        result[index_to_replace] = 1
    return result

def predict_method_crashing_chance(pairs):
    # Introduce a 50% chance of crashing
    if random.random() < 0.5:  # random.random() generates a float between 0 and 1
        raise Exception("Random crash occurred!")

    # Simulate processing time
    time.sleep(0.005 * len(pairs))
    return [True for _ in pairs]







if __name__ == '__main__':
    # example usage
    #
    #     the value of this dict, is the predict function for a method,
    #     given a batch and some positive and negative icl examples.
    # inputs are:
    # List[Tuple[str, str]]: the given batch (two sentences for classification)
    #
    # expected output:
    # List[int] list of classifications
    # out[0] means the sentences in in[0] (first element of the given batch) aren't paraphrases
    methods: Dict[
        str,
        Callable[[List[Tuple[str, str]]], List[bool]]
    ] = {
        "m1": predict_method1,
        "m2": predict_method2
    }
    bench(methods, dataset_keys=None, bench_id="initial"
          , purge_results=True
          )
