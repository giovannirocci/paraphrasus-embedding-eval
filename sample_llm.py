from openai import OpenAI
from benchmarking import bench
from logger import mylog
import pandas as pd
import sys

logger = mylog.get_logger()



client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
prompt_type_to_question = {
        "(Paraph)": "paraphrases",
        "(Sem Equiv)": "semantically equivalent",
        "(Ex. Same Content)": "expressing the same content",
    }

def predict_llama3_p1(batch):
    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)


def predict_llama3_p2(batch):
    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Sem Equiv)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)


def predict_llama3_p3(batch):
    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Ex. Same Content)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)

def predict_llama3_iclk4_p1(batch):
    all_samples = pd.read_csv("icl_samples/icl_samples_paws_x.tsv", sep="\t")
    icl_positive_samples = all_samples[all_samples["label"] == 1]
    icl_negative_samples = all_samples[all_samples["label"] == 0]

    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question, icl_negative_samples, icl_positive_samples, icl_k=4)


def predict_llama3_iclk4_p2(batch):
    all_samples = pd.read_csv("icl_samples/icl_samples_paws_x.tsv", sep="\t")
    icl_positive_samples = all_samples[all_samples["label"] == 1]
    icl_negative_samples = all_samples[all_samples["label"] == 0]

    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Sem Equiv)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question, icl_negative_samples, icl_positive_samples,
                                                   icl_k=4)

def predict_llama3_iclk4_p3(batch):
    all_samples = pd.read_csv("icl_samples/icl_samples_paws_x.tsv", sep="\t")
    icl_positive_samples = all_samples[all_samples["label"] == 1]
    icl_negative_samples = all_samples[all_samples["label"] == 0]

    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Ex. Same Content)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question, icl_negative_samples, icl_positive_samples,
                                                   icl_k=4)

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

def content_from_sentences(s1,s2):
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return f"Are the following sentences {question}?\n\nSentence 1: {s1}\nSentence 2: {s2}\n\nAnswer with 'Yes' or 'No'"


def predict_qwen_qwq_32b_prev_q8(batch):
    model = "qwen_qwq-32b-preview_mlx"
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)

def predict_llama3_3_70b_ins_8bit(batch):
    model = "llama-3.3-70b-instruct"
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)

def predict_llama3_8b_ins_4bit(batch):
    model = "meta-llama-3-8b-instruct"
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)

def predict_llama3_8b_ins_8bit(batch):
    model = "llama-3-8b-instruct-1048k"
    prompt_type = "(Paraph)"
    question = prompt_type_to_question[prompt_type]
    return predict_paraphrases_in_batches_lmstudio(model, batch, question)

wrongs = []

def predict_paraphrases_in_batches_lmstudio(
    model, batch, question, icl_negative_samples=None, icl_positive_samples=None, icl_k=None
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
    # print(sum(paraphrase_predictions))
    return paraphrase_predictions

methods_from_paper = {
    "llama3_p1": predict_llama3_p1,
    "llama3_p2": predict_llama3_p2,
    "llama3_p3": predict_llama3_p3,
    "llama3_icl_k4_p1": predict_llama3_iclk4_p1,
    "llama3_icl_k4_p2": predict_llama3_iclk4_p2,
    "llama3_icl_k4_p3": predict_llama3_iclk4_p3
}

custom_methods ={
            "llama_3_3_70b_ins_q8": predict_llama3_3_70b_ins_8bit,
            'llama_3_8b_ins_q4_k_m': predict_llama3_8b_ins_4bit
    }

if __name__ == '__main__':
    if len(sys.argv) == 1:
        bench_id = "reproduce"
    elif len(sys.argv) == 2:
        bench_id = sys.argv[1]
    else:
        print("Please only specify the bench identifier!")
        exit(1)

    bench(
        methods=methods_from_paper,
    bench_id=bench_id,
    batch_size=1,
    )