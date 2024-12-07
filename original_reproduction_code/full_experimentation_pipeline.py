import os
import json
import argparse
import pandas as pd
from collections import Counter
from functools import partial
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import pipeline
import torch
from openai import OpenAI
import numpy as np


# Function to check if pairs are paraphrases in batches using a zero-shot classifier
def zeroshot_encoder_predictions(zero_shot_classifier, batch, candidate_labels):
    """
    Predicts if sentence pairs in a batch are paraphrases using a zero-shot classifier.

    Parameters:
        zero_shot_classifier (function): The zero-shot classifier function.
        batch (list of tuples): List of sentence pairs.
        hypothesis_template (str): Hypothesis template for the classifier.
        classes_verbalized (list of str): List of verbalized classes for the classifier.

    Returns:
        list: Predictions indicating if each pair is a paraphrase.
    """
    texts = []
    for sample in batch:
        text1 = f"{sample[0]} {sample[1]}"
        texts.append(text1)
    outputs = zero_shot_classifier(texts, candidate_labels, multi_label=False)
    labels_map = {k: 0 for k in candidate_labels}
    labels_map[candidate_labels[0]] = 1
    predictions = [labels_map[output["labels"][0]] for output in outputs]
    return predictions


# Function to save dataset to a CSV file in the backup directory
def save_backup_dataset(dataset, path, run_name):
    """
    Saves the dataset to a backup directory with a modified filename.

    Parameters:
        dataset (pd.DataFrame): The dataset to save.
        path (str): The original file path.
        run_name (str): The run name to include in the backup filename.

    Returns:
        None
    """
    # Create the backup directory if it doesn't exist
    backup_dir = os.path.join(os.path.dirname(path), "backup")
    os.makedirs(backup_dir, exist_ok=True)

    # Modify path to create backup filename with run name
    base_filename = os.path.basename(path)[:-4] + f"_{run_name}_backup.tsv"
    backup_path = os.path.join(backup_dir, base_filename)

    # Save the DataFrame to a CSV file
    dataset.to_csv(backup_path, index=False, sep="\t")
    print(f"Dataset saved to {backup_path}")


# Function to prepare data batches
def prepare_batches(data, batch_size=64):
    """
    Prepares batches of data.

    Parameters:
        data (list): The list of data items.
        batch_size (int): The size of each batch.

    Returns:
        list of lists: Batches of data.
    """
    batches = []
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


sanity_check_predictions = []


# Function to prepare in-context learning (ICL) examples
def prepare_prepend_icl(
    icl_negative_samples, icl_positive_samples, icl_k, question
):  # update to include question variable
    """
    Prepares a prompt with in-context learning (ICL) examples.

    Parameters:
        icl_negative_samples (pd.DataFrame): Negative samples for ICL.
        icl_positive_samples (pd.DataFrame): Positive samples for ICL.
        icl_k (int): Number of ICL examples (should be even).
        question: The phrasing of paraphrase for the question

    Returns:
        str: The ICL prompt string.
    """
    # Ensure icl_k is even, binary classification
    positives = icl_positive_samples.sample(int(icl_k / 2)).reset_index(drop=True)
    negatives = icl_negative_samples.sample(int(icl_k / 2)).reset_index(drop=True)

    # Initialize the prompt string
    prepend_prompt = f"Here are examples sentence pairs that are {question} (Yes) and not {question} (No):\n\n"

    # Alternate between positive and negative samples
    for i in range(int(icl_k / 2)):
        # Add positive example
        prepend_prompt += f"Sentence 1: {positives.loc[i, 'sentence1']}\nSentence 2: {positives.loc[i, 'sentence2']}\nYes\n\n"
        # Add negative example
        prepend_prompt += f"Sentence 1: {negatives.loc[i, 'sentence1']}\nSentence 2: {negatives.loc[i, 'sentence2']}\nNo\n\n"

    return prepend_prompt


# Function to check if pairs are paraphrases using OpenAI API
def predict_paraphrases_in_batches_llama(
    batch, question, icl_negative_samples=None, icl_positive_samples=None, icl_k=None
):
    """
    Predicts if sentence pairs in a batch are paraphrases using the OpenAI API with optional ICL.

    Parameters:
        batch (list of tuples): List of sentence pairs.
        question: The phrasing of paraphrase for the question
        icl_negative_samples (pd.DataFrame, optional): Negative samples for ICL. Defaults to None.
        icl_positive_samples (pd.DataFrame, optional): Positive samples for ICL. Defaults to None.
        icl_k (int, optional): Number of ICL examples. Defaults to None.

    Returns:
        list: Predictions indicating if each pair is a paraphrase.
    """
    paraphrase_predictions = []
    for sentence1, sentence2 in batch:
        if icl_positive_samples is None or icl_negative_samples is None:
            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You are a helpful Assistant."},
                    {
                        "role": "user",
                        "content": f"Are the following sentences {question}?\n\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer with 'Yes' or 'No'",
                    },
                ],
                max_tokens=3,
            )
        else:
            prepend_icl = prepare_prepend_icl(
                icl_negative_samples, icl_positive_samples, icl_k, question
            )
            completion = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You are a helpful Assistant."},
                    {
                        "role": "user",
                        "content": prepend_icl
                        + f"Are the following sentences {question}?\n\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer with 'Yes' or 'No'",
                    },
                ],
                max_tokens=3,
            )

        answer = completion.choices[0].message.content
        sanity_check_predictions.append(answer[:3])
        paraphrase_predictions.append(int(answer[:3].lower() == "yes"))
    print(sum(paraphrase_predictions))
    return paraphrase_predictions


# Function to tokenize sentence pairs using a tokenizer
def tokenize_function(batch, tokenizer):
    """
    Tokenizes sentence pairs using the specified tokenizer.

    Parameters:
        batch (list of tuples): List of sentence pairs.
        tokenizer (function): The tokenizer function.

    Returns:
        dict: Tokenized output.
    """
    # Extract sentences from the batch
    sentence1, sentence2 = zip(*batch)

    # Tokenize all sentence pairs at once
    tokenized_output = tokenizer(
        list(sentence1),
        list(sentence2),
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    return tokenized_output


def main(config_file):
    # Load configuration from a JSON file
    with open(config_file) as file:
        config = json.load(file)

    # Determine the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():  # Check for MPS availability
        device = "mps"
    else:
        device = "cpu"

    print(device)
    dataset_pipeline = {
        "stsbenchmark": "datasets/stsbenchmark-test-sts.tsv",
        "ms_mrpc": "datasets/ms-mrpc.tsv",
        "onestop_all": "datasets/onestop_parallel_all_pairs.tsv",
        "simple_amr": "datasets/amr_true_paraphrases.tsv",
        "fb_anli_pre_hyp": "datasets/fb-anli-pre-hyp.tsv",
        "fb_anli_hyp_pre": "datasets/fb-anli-hyp-pre.tsv",
        "sickr_sts": "datasets/sickr-sts.tsv",
        "pawsx_test": "datasets/paws-x-test.tsv",
        "stanfordnlp_snli_pre_hyp": "datasets/stannlp-snli-pre-hyp.tsv",
        "fb_xnli_pre_hyp": "datasets/fb-xnli-pre-hyp.tsv",
        "fb_xnli_hyp_pre": "datasets/fb-xnli-hyp-pre.tsv",
        "stanfordnlp_snli_hyp_pre": "datasets/stannlp-snli-hyp-pre.tsv",
    }
    datasets = config.get("datasets", "all")  # Default to "all" if not specific
    datasets_to_run = (
        list(dataset_pipeline.keys())
        if config["datasets"] == "all"
        else config["datasets"]
    )

    run_name = config["run_name"]
    method = config[
        "method"
    ]  # possible methods are: ["LLM zero-shot", "LLM icl_K", "LLM trained", "Encoder trained", "Encoder zero-shot"]
    model_path = config.get("model_path", None)
    prompt_type = config.get("prompt_type", "(Paraph)")
    batch_size = config.get("batch_size", 32)  # Default to 32 if not specified

    prompt_type_to_question = {
        "(Paraph)": "paraphrases",
        "(Ex. Same Content)": "expressing the same content",
        "(Sem Equiv)": "semantically equivalent",
    }

    if method.startswith("LLM "):
        # Example: reuse your existing OpenAI setup
        global client
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        question = prompt_type_to_question[prompt_type]
    if method.startswith("LLM icl_"):
        icl_k = int(method.split("_")[-1])  # Extract the specific icl_samples if needed
        all_samples = pd.read_csv("datasets/icl_samples_paws_x.tsv", sep="\t")
        icl_positive_samples = all_samples[all_samples["label"] == 1]
        icl_negative_samples = all_samples[all_samples["label"] == 0]
        question = prompt_type_to_question[prompt_type]
    if method.startswith("Encoder trained"):
        encoder_tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        encoder_model = XLMRobertaForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )
        encoder_model.to(device)
        # Use functools.partial to create a new function that has the tokenizer included
        tokenize_with_tokenizer = partial(
            tokenize_function, tokenizer=encoder_tokenizer
        )
    if method.startswith("Encoder zero-shot"):
        candidate_labels = ["paraphrase", "related", "unrelated"]
        paraphrase_zs_classifier = pipeline(
            "zero-shot-classification", model=model_path, device=device
        )

    # Loop through the dataset pipeline and process datasets
    for shortname in datasets_to_run:
        print(shortname)
        dataset_path = dataset_pipeline[shortname]
        # Load the dataset
        dataset = pd.read_csv(dataset_path, sep="\t")
        print(len(dataset))
        if run_name in dataset:
            print(
                run_name
                + " is in the dataset "
                + dataset_path
                + ". Skipping prediction"
            )
            continue
        # Save the dataset to a backup CSV file in the backup directory
        save_backup_dataset(dataset, dataset_path, run_name)
        prediction_pairs = [
            (example["sentence1"], example["sentence2"])
            for example in dataset.to_dict("records")
        ]
        batches = prepare_batches(prediction_pairs)
        all_predictions = []
        for batch in batches:
            if method == "LLM zero-shot":
                predictions = predict_paraphrases_in_batches_llama(batch, question)
            elif method.startswith("LLM icl_"):  # Fixed syntax for method check
                predictions = predict_paraphrases_in_batches_llama(  # update with implementation of question
                    batch,
                    question,
                    icl_positive_samples=icl_positive_samples,
                    icl_negative_samples=icl_negative_samples,
                    icl_k=icl_k,
                )
            elif method == "LLM trained":
                # Implementation to be added later. I used COLAB for now.
                pass
            elif method == "Encoder trained":
                # Assuming tokenize_with_tokenizer is a function that takes a batch and returns tokenized data
                tokenized_batch = tokenize_with_tokenizer(batch)
                # Assuming tokenized_batch is already a dictionary-like object of tensors
                tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
                with torch.no_grad():
                    outputs = encoder_model(**tokenized_batch)
                    logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).tolist()
            elif method == "Encoder zero-shot":
                predictions = zeroshot_encoder_predictions(
                    paraphrase_zs_classifier, batch, candidate_labels
                )
            else:  # unspecified method
                raise ValueError(f"Invalid method specified: {method}")

            all_predictions.extend(predictions)
        dataset[run_name] = all_predictions
        dataset.to_csv(dataset_path, index=False, sep="\t")
    print(Counter(sanity_check_predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dataset processing and prediction."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration JSON file."
    )
    args = parser.parse_args()
    main(args.config_file)
