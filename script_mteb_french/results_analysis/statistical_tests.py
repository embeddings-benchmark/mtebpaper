import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser, Namespace

from results_parser import ResultsParser


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--results_folder", required=True, type=str)
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./script_mteb_french/results_analysis/statistical_tests_results",
    )
    args = parser.parse_args()

    return args


def run_statistical_tests(data: pd.DataFrame, output_path: str):
    results_lists = list(data.fillna(0).values[:, 1:])
    friedman_stats = friedmanchisquare(*results_lists)
    print(f"Running friedman test on {len(results_lists)} models...")
    if friedman_stats.pvalue < 0.05:
        print(
            f"There is a significant difference between the models (p-value: {friedman_stats.pvalue}). Running post-hoc tests..."
        )
        data_melted = data.melt(id_vars="model", var_name="dataset", value_name="score")
        data_melted = data_melted.dropna(axis=0, how='any')
        avg_rank = (
            data_melted.groupby("dataset")
            .score.rank(pct=True, ascending=False)
            .groupby(data_melted.model)
            .mean()
        )
        detailed_test_results = sp.posthoc_conover_friedman(
            data_melted,
            melted=True,
            block_col="dataset",
            group_col="model",
            y_col="score",
        )
        plt.figure(figsize=(10, 8))
        plt.title("Post hoc conover friedman tests")
        sp.sign_plot(detailed_test_results)
        plt.savefig(
            os.path.join(output_path, "conover_friedman.png"), bbox_inches="tight"
        )
        plt.figure(figsize=(10, 6))
        plt.title("Critical difference diagram of average score ranks")
        sp.critical_difference_diagram(avg_rank, detailed_test_results)
        plt.savefig(
            os.path.join(output_path, "critical_difference_diagram.png"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    args = parse_args()
    rp = ResultsParser()
    results_df = rp(args.results_folder, return_main_scores=False)
    results_df = results_df.droplevel(0, axis=1)
    results_df = results_df.reset_index()
    results_df["model"] = results_df["model"].apply(
        lambda x: x.replace(args.results_folder, "")
    )
    run_statistical_tests(results_df, args.output_folder)
