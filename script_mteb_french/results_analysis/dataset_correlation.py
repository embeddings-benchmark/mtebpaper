import json
import os
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


if __name__ == "__main__":

    args = parse_args()
    # Get results
    rp = ResultsParser()
    results_df = rp(args.results_folder, return_main_scores=False)
    results_df = results_df.droplevel(0, axis=1)
    results_df.index = results_df.index.map(
        lambda x: x.replace(args.results_folder, "")
    )
    print(results_df.shape)
    # Prepare output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    # Dataset correlations
    spearman_corr_matrix_datasets = results_df.corr(method="spearman")
    spearman_corr_matrix_datasets.to_csv(
        os.path.join(args.output_folder, "spearman_corr_matrix_datasets.csv")
    )
    plt.figure(figsize=(10, 8))
    mask = np.tril(np.ones_like(spearman_corr_matrix_datasets, dtype=bool))
    spearman_corr_matrix_datasets = (
        spearman_corr_matrix_datasets.fillna(0) * mask
    ).map(lambda x: np.nan if x == 0 else x)
    sns.heatmap(
        spearman_corr_matrix_datasets * mask, fmt=".2f", linewidths=0.5, cmap="coolwarm"
    )
    plt.title("Dataset Correlation Heatmap (Spearman)")
    plt.savefig(
        os.path.join(args.output_folder, "spearman_corr_heatmap_datasets.png"),
        bbox_inches="tight",
    )
    # Model correlations
    transposed_results_df = results_df.transpose()
    print(transposed_results_df)
    spearman_corr_matrix_models = transposed_results_df.corr(method="spearman")
    spearman_corr_matrix_models.to_csv(
        os.path.join(args.output_folder, "spearman_corr_matrix_models.csv")
    )
    plt.figure(figsize=(18, 15))
    mask = np.tril(np.ones_like(spearman_corr_matrix_models, dtype=bool))
    spearman_corr_matrix_models = (spearman_corr_matrix_models.fillna(0) * mask).map(
        lambda x: np.nan if x == 0 else x
    )
    sns.heatmap(spearman_corr_matrix_models, fmt=".2f", linewidths=0.5, cmap="coolwarm")
    plt.title("Model Correlation Heatmap (Spearman)")
    plt.savefig(
        os.path.join(args.output_folder, "spearman_corr_heatmap_models.png"),
        bbox_inches="tight",
    )
