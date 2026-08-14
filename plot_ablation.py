import argparse
import json
import os
import matplotlib.pyplot as plt


def plot_results(res_dir, save_path):
    """
    Plots the results for the three model with the different calibration sets sizes.

    Args:
        res_dir (str): The directory containing the result files.
        save_path (str): The path to save the plot.
    """
    model_names = ["mE5-instr", "Qwen3-Emb", "M-MiniLM"]
    model_colors = {
        "mE5-instr": "tab:blue",
        "Qwen3-Emb": "tab:orange",
        "M-MiniLM": "tab:green",
    }

    # Keyed by calibration size (int) -> value, so alignment doesn't depend
    # on the (arbitrary) order os.listdir() returns entries in.
    lr_results = {name: {} for name in model_names}
    thr_results = {name: {} for name in model_names}

    mapping = {
                "multilingual-e5-large-instruct": "mE5-instr",
                "Qwen3-Embedding-0.6B": "Qwen3-Emb",
                "paraphrase-multilingual-MiniLM-L12-v2": "M-MiniLM",
            }

    for dir in os.listdir(res_dir):
        dir_path = os.path.join(res_dir, dir)
        if not os.path.isdir(dir_path):
            continue
        try:
            size = int(dir)
        except ValueError:
            print(f"Skipping non-numeric directory: {dir}")
            continue

        for filename in os.listdir(dir_path):
            if not filename.endswith('.json'):
                continue
            model_key = filename.split('_')[1]
            if model_key not in mapping:
                print(f"Skipping unrecognized model file: {filename}")
                continue
            model_name = mapping[model_key]
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                lr_results[model_name][size] = data['classifier_error']['overall_mean']
                thr_results[model_name][size] = data['threshold_error']['overall_mean']

    # Convert the size -> value dicts into sorted, aligned x/y lists.
    for results in (lr_results, thr_results):
        for name, values in results.items():
            sizes = sorted(values.keys())
            results[name] = {'x': sizes, 'y': [values[s] for s in sizes]}

    plt.figure(figsize=(20, 12))
    for key, value in lr_results.items():
        if value['x']:
            plt.plot(
                value['x'],
                value['y'],
                marker='o',
                linestyle='-',
                linewidth=4,
                color=model_colors[key],
                label=f"{key} - LR.",
            )
    for key, value in thr_results.items():
        if value['x']:
            plt.plot(
                value['x'],
                value['y'],
                marker='o',
                linestyle='--',
                linewidth=4,
                color=model_colors[key],
                label=f"{key} - Thr.",
            )

    plt.xlabel('Calibration Set Size (per dataset)', fontsize=26)
    plt.ylabel('Avg. Error', fontsize=26)
    plt.title('Performance of Models for Different Calibration Set Sizes', fontsize=32, fontweight='bold', pad=20)
    plt.legend(fontsize=26, loc='upper right')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results for different calibration set sizes.")
    parser.add_argument('--res_dir', type=str, help='Directory containing the result files.', default='ablation')
    parser.add_argument('--save_path', type=str, help='Path to save the plot.', default='plots/calibration_size_ablation.pdf')
    args = parser.parse_args()

    plot_results(args.res_dir, args.save_path)