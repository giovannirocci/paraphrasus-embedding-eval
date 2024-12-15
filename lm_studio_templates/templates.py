from openai import OpenAI
from benchmarking import bench
from logger import mylog
import pandas as pd
import sys

logger = mylog.get_logger()



local_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
remote_client = OpenAI(base_url="http://192.168.1.176:1234/v1", api_key="lm-studio")

def predict_lm_studio(client, batch, model: str, p: int, icl=False, icl_alt=False):
    all_samples = pd.read_csv("original_reproduction_code/datasets/icl_samples_paws_x.tsv", sep="\t")
    icl_positive_samples = all_samples[all_samples["label"] == 1] if icl else None
    icl_negative_samples = all_samples[all_samples["label"] == 0] if icl else None

    if p == 1:
        prompt_type = "(Paraph)"
    elif p == 2:
        prompt_type = "(Sem Equiv)"
    elif p == 3:
        prompt_type = "(Ex. Same Content)"
    else:
        logger.error(f"Unexpected question type: {p}! p should only come from [1,3]!")
        exit(1)

    prompt_type_to_question = {
        "(Paraph)": "paraphrases",
        "(Sem Equiv)": "semantically equivalent",
        "(Ex. Same Content)": "expressing the same content",
    }

    question = prompt_type_to_question[prompt_type]

    return predict_paraphrases_in_batches_lmstudio(client, model, batch, question, icl_negative_samples, icl_positive_samples,
                                                   icl_k=4, use_alt_icl=icl_alt)

def predict_paraphrases_in_batches_lmstudio(
    client, model, batch, question, icl_negative_samples=None, icl_positive_samples=None, icl_k=None, use_alt_icl=False
):
    """
    Predicts if sentence pairs in a batch are paraphrases using the OpenAI API with optional ICL.

    Parameters:
        batch (list of tuples): List of sentence pairs.
        question: The phrasing of paraphrase for the question

    Returns:
        list: Predictions indicating if each pair is a paraphrase.
    """
    paraphrase_predictions = []
    for sentence1, sentence2 in batch:
        if icl_positive_samples is None or icl_negative_samples is None:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful Assistant."},
                    {
                        "role": "user",
                        "content": f"Are the following sentences {question}?\n\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer with 'Yes' or 'No'",
                    },
                ],
                max_tokens=3,
                # temperature=0,
            )
        elif use_alt_icl:
            messages = get_messages_with_icl_alt(icl_negative_samples=icl_negative_samples, icl_positive_samples=icl_positive_samples, icl_k=icl_k, question=question)
            messages.append({
                "role": "user",
                "content": f"Are the following sentences {question}?\n\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer with 'Yes' or 'No'"
            })
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=3,
            )
        else:
            prepend_icl = prepare_prepend_icl(
                icl_negative_samples, icl_positive_samples, icl_k, question
            )
            completion = client.chat.completions.create(
                model=model,
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

        # sanity_check_predictions.append(answer)

        answer = completion.choices[0].message.content

        if 'yes' in answer.lower():
            paraphrase_predictions.append(True)
        elif 'no' in answer.lower():
            paraphrase_predictions.append(False)
        else:
            wrongs.append(answer)
            logger.info(f"Wrong answers: {", ".join(wrongs)}")
            return []
    return paraphrase_predictions



def get_messages_with_icl_alt(
    icl_negative_samples, icl_positive_samples, icl_k, question
):

    def format_question(sentence1, sentence2):
        return f"Are the following sentences {question}?\n\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n\nAnswer with 'Yes' or 'No'"

    messages = [
        {"role": "system", "content": "You are a helpful Assistant."}
    ]

    # Ensure icl_k is even, binary classification
    positives = icl_positive_samples.sample(int(icl_k / 2)).reset_index(drop=True)
    negatives = icl_negative_samples.sample(int(icl_k / 2)).reset_index(drop=True)

    # Alternate between positive and negative samples
    for i in range(int(icl_k / 2)):
        # Add positive example
        s1 = positives.loc[i, 'sentence1']
        s2 = positives.loc[i, 'sentence2']

        messages.append({
            "role": "user",
            "content": format_question(s1, s2)
        })
        messages.append({
            "role": "assistant",
            "content": "Yes"
        })

        # Add negative example

        s1 = negatives.loc[i, 'sentence1']
        s2 = negatives.loc[i, 'sentence2']

        messages.append({
            "role": "user",
            "content": format_question(s1, s2)
        })
        messages.append({
            "role": "assistant",
            "content": "No"
        })

    return messages
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





wrongs = []


