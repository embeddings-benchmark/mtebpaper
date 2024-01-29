import json
import os
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
import seaborn as sns

from results_parser import ResultsParser

def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--results_folder", required=True, type=str)
    parser.add_argument("--output_folder", type=str, default="./correlation_analysis")
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    # Get results
    rp = ResultsParser()
    results_df, tasks_main_scores_subset = rp(args.results_folder, return_main_scores=True)
    # Prepare output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    # Dataset correlations
    spearman_corr_matrix_datasets = results_df.corr(method='spearman')
    spearman_corr_matrix_datasets.to_csv(os.path.join(args.output_folder, "spearman_corr_matrix_datasets.csv"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix_datasets, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Dataset Correlation Heatmap (Spearman)')
    plt.savefig(os.path.join(args.output_folder, "spearman_corr_heatmap_datasets.png"), bbox_inches='tight')
    with open(os.path.join(args.output_folder, "main_scores.json"), 'w') as f:
        json.dump(tasks_main_scores_subset, f, indent=4)
    # Model correlations
    transposed_results_df = results_df.transpose()
    spearman_corr_matrix_models = transposed_results_df.corr(method='spearman')
    spearman_corr_matrix_models.to_csv(os.path.join(args.output_folder, "spearman_corr_matrix_models.csv"))
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr_matrix_models, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Model Correlation Heatmap (Spearman)')
    plt.savefig(os.path.join(args.output_folder, "spearman_corr_heatmap_models.png"), bbox_inches='tight')
