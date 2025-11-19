import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "embedding_benchmarks/balanced"  # adjust to your directory
OUT_PATH = "plots/overall_auc.png"


def load_overall_auc(results_dir):
    model_names = []
    aucs = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Try to get AUC in a robust way
        if "overall_auc" in data:
            auc = data["overall_auc"]
        else:
            raise ValueError(f"No AUC score found in {fname}")

        # model name from filename (strip method/flags etc. if you want)
        pattern = r'_(elementwise_diff|multiplication|sum)(_(auc|error|full_results))?\.json$'
        path = re.sub(pattern, '', path)
        model_name = path.split('/')[-1].split('_')[-1]

        model_names.append(model_name)
        aucs.append(auc)

    return model_names, aucs


def plot_overall_auc(model_names, aucs, title="Overall AUC per model", out_path=None):
    # sort by AUC
    sorted_idx = np.argsort(aucs)
    model_names = [model_names[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(model_names, aucs, color='cornflowerblue')
    plt.xlabel("AUC")
    plt.title(title)
    plt.xlim(0.5, 1.0)  # adjust if you want
    for i, v in enumerate(aucs):
        plt.text(v + 0.001, i, f"{v:.3f}", va="center")
    plt.tight_layout()

    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    model_names, aucs = load_overall_auc(RESULTS_DIR)
    plot_overall_auc(model_names, aucs, out_path=OUT_PATH)
