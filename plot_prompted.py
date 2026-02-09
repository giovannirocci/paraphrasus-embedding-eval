import os
import json
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_data_from_dir(prompted_dir, baseline_path, metric):
    prompted_files = [f for f in os.listdir(prompted_dir) if f.endswith('.json')]
    data = []

    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)

    data.append({
        'Prompt': 'Standard',
        'Classify': baseline_data[metric]['overall_classify'],
        'Minimize': baseline_data[metric]['overall_minimize'],
        'Maximize': baseline_data[metric]['overall_maximize'],
        'Overall': baseline_data[metric]['overall_mean']
    })
    
    for pf in prompted_files:
        with open(os.path.join(prompted_dir, pf), 'r') as f:
            prompted_data = json.load(f)
        
        if metric in prompted_data:
            label = ""
            if pf.startswith('p1'): label = 'P1 (Paraphrase)'
            elif pf.startswith('p2'): label = 'P2 (Semantic Equivalent)'
            elif pf.startswith('p3'): label = 'P3 (Express Same Content)'
            
            if label:
                data.append({
                    'Prompt': label,
                    'Classify': prompted_data[metric]['overall_classify'],
                    'Minimize': prompted_data[metric]['overall_minimize'],
                    'Maximize': prompted_data[metric]['overall_maximize'],
                    'Overall': prompted_data[metric]['overall_mean']
                })
    return pd.DataFrame(data)

def plot_combined_results(prompted_dir, baseline_path, output_path):
    # Load data for both metrics
    df_threshold = get_data_from_dir(prompted_dir, baseline_path, 'threshold_error')
    df_classifier = get_data_from_dir(prompted_dir, baseline_path, 'classifier_error')
    
    avg_types = ['Classify', 'Minimize', 'Maximize', 'Overall']
    short_labels = {'Standard': 'Standard', 'P1 (Paraphrase)': 'P1', 
                    'P2 (Semantic Equivalent)': 'P2', 'P3 (Express Same Content)': 'P3'}
    
    df_threshold['Short Prompt'] = df_threshold['Prompt'].map(short_labels)
    df_classifier['Short Prompt'] = df_classifier['Prompt'].map(short_labels)

    # Create a 4x2 grid (4 metrics x 2 calibration types)
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True, sharey=True)
    
    # Adjust spacing for "Density"
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    col_titles = ["Threshold Calibration", "Classifier Calibration"]
    dataframes = [df_threshold, df_classifier]

    for col in range(2):
        df = dataframes[col]
        axes[0, col].set_title(col_titles[col], fontsize=14, fontweight='bold', pad=20)
        
        for row, avg in enumerate(avg_types):
            ax = axes[row, col]
            sns.barplot(data=df, x='Short Prompt', y=avg, ax=ax, width=0.6, hue='Prompt', palette='Set2', legend="brief" if  (row, col)==(1,0) else False)
            
            if (row, col) == (1, 0):
                ax.legend(title="Multilingual E5-Instruct Prompt", loc='upper left')

            # Subtitles for each row (only center-ish or on each)
            if col == 0:
                ax.set_ylabel("Avg. Error", fontsize=10)
            else:
                ax.set_ylabel("")

            # Set row titles (Subtitles)
            ax.text(0.5, 1.02, avg, transform=ax.transAxes, ha='center', fontsize=11)
            
            # Styling
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_xlabel("")
            ax.set_ylim(0, 0.55)
            
            # Value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=2, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompted_dir', type=str, default='embedding_benchmarks/prompted')
    parser.add_argument('--baseline_path', type=str, default='embedding_benchmarks/comparable/intfloat_multilingual-e5-large-instruct_comparable_full_results.json')
    parser.add_argument('--output_path', type=str, default='plots/prompt_errors.png')
    args = parser.parse_args()

    plot_combined_results(args.prompted_dir, args.baseline_path, args.output_path)