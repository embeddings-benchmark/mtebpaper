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
        default="./analyses_outputs/statistical_tests",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
    )
    args = parser.parse_args()

    return args


def run_statistical_tests(data: pd.DataFrame, output_path: str, output_format:str = "pdf"):
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
        plt.figure(figsize=(12, 15))
        plt.rcParams.update({'font.size': 15})
        plt.title("Post hoc conover friedman tests")
        sp.sign_plot(detailed_test_results)
        plt.savefig(
            os.path.join(output_path, f"conover_friedman.{output_format}"), bbox_inches="tight"
        )
        plt.figure(figsize=(12, 8))
        sp.critical_difference_diagram(avg_rank, detailed_test_results)
        plt.savefig(
            os.path.join(output_path, f"critical_difference_diagram.{output_format}"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    rp = ResultsParser()
    results_df = rp(args.results_folder, return_main_scores=False)
    results_df = results_df.droplevel(0, axis=1)
    results_df = results_df.reset_index()
    results_df["model"] = results_df["model"].apply(
        lambda x: os.path.basename(x)
    )
    run_statistical_tests(results_df, args.output_folder, args.output_format)
