from datasets import load_dataset
import random
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--min_size", type=int, default=3, help="Minimum size of paraphrase sets to keep")
parser.add_argument("--max_size", type=int, default=20, help="Maximum size of paraphrase sets to keep")
parser.add_argument("--sample_n", type=int, help="If set, sample this many sets")
parser.add_argument("--out", type=str,
                    default="datasets_no_results/tapaco_paraphrases.json", help="Output JSON file path to save the dataset")
args = parser.parse_args()

random.seed(42)

def filter_sets(sets, min_size, max_size):
    filtered = {}
    for k, v in sets.items():
        if len(v) < min_size:
            continue
        elif len(v) > max_size:
            filtered[k] = random.sample(v, max_size)
        else:
            filtered[k] = v
    return filtered

def sample_subset(sets, n):
    keys = list(sets.keys())
    sampled_keys = random.sample(keys, n)
    return {k: sets[k] for k in sampled_keys}

def pairify(sets):
    data = {}
    for n, v in enumerate(sets.values()):
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                data[f"tapaco/{n}_{i}_{j}"] = {"sentence1": v[i], "sentence2": v[j]}
    return data

def output_json(sets, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sets, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print("Loading Tapaco dataset...")
    ds = load_dataset("community-datasets/tapaco", "all_languages", split="train")
    df = ds.to_pandas()

    grouped = df.groupby("paraphrase_set_id")
    sentences = grouped["paraphrase"].apply(list).to_dict()

    print(f"Filtering paraphrase sets... (min_size={args.min_size}, max_size={args.max_size})")
    sent = filter_sets(sentences, args.min_size, args.max_size)

    if args.sample_n:
        print(f"Sampling {args.sample_n} paraphrase sets...")
        sent = sample_subset(sent, args.sample_n)

    pairs = pairify(sent)
    output_json(pairs, args.out)
    print(f"Saved {len(pairs)} pairs to {args.out}")
