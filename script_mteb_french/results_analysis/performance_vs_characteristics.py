import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser, Namespace

from results_parser import ResultsParser
import numpy as np

# model,pretrained_or_tuned,multilingual_or_french,number_params,size_gb,seq_len,embedding_dim,model_type,license
CHARACTERISTICS = {
    "finetuned": "numerical",
    "multilingual_or_french": "categorical",
    "number_params": "numerical",
    "size_gb": "numerical",
    "seq_len": "numerical",
    "embedding_dim": "numerical",
    "model_type": "categorical",
    "license": "categorical",
    "tuned_on_sentence_sim": "numerical",
}

CHARACTERISTICS_DISPLAY_NAMES = {
    "finetuned": "Finetuned vs pretrained",
    "multilingual_or_french": "Language",
    "number_params": "Model number of parameters",
    "size_gb": "Model size",
    "seq_len": "Max sequence length",
    "embedding_dim": "Embedding dimension",
    "model_type": "Model type",
    "license": "License",
    "tuned_on_sentence_sim": "Tuned for sentence similarity",
}

COLS_TO_KEEP_GLOBAL_CORRELATION = {
    "score": "Model ranking",
    "finetuned": "Finetuned vs pretrained",
    "number_params": "Model number of parameters",
    "seq_len": "Max sequence length",
    "embedding_dim": "Embedding dimension",
    "tuned_on_sentence_sim": "Tuned for sentence similarity",
    "bilingual": "Bilingual",
    "english": "English",
    "english_plus": "English +\ntuning on\nother languages",
    "french": "French",
    "multilingual": "Multilingual",
    "Closed source": "Closed source",
}


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--results_folder", required=True, type=str)
    parser.add_argument("--characteristics_csv", required=True, type=str)
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./analyses_outputs/performance_vs_characteristics",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
    )
    args = parser.parse_args()

    return args


def prepare_data(
    results_df: pd.DataFrame, characteristics_df: pd.DataFrame, mode: str = "avg"
):
    data = results_df.assign(**results_df.iloc[:, 1:].rank(pct=True))
    data = data.melt(id_vars="model", var_name="dataset", value_name="score")
    data = data[["model", "score"]]
    if mode == "avg":
        data = data.groupby("model").mean().reset_index()
    data = data.merge(characteristics_df, on="model", how="left")

    return data


def global_correlation(
    results_df: pd.DataFrame, characteristics_df: pd.DataFrame,
    output_path: str, output_format:str="pdf",
):
    data = prepare_data(results_df, characteristics_df, mode="avg")
    data = data.drop(columns=["model"])
    # get dummies for categorical variables
    data = pd.get_dummies(data, prefix="", prefix_sep="")
    data = data[list(COLS_TO_KEEP_GLOBAL_CORRELATION.keys())]
    data = data.rename(columns=COLS_TO_KEEP_GLOBAL_CORRELATION)
    # compute correlation matrix
    corr_matrix = data.corr(method="spearman")
    corr_matrix.map(lambda x:round(x,3)).to_csv(os.path.join(output_path, "correlation_matrix.csv"))
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix = corr_matrix.where(mask)
    # plot correlation heatmap
    plt.figure(figsize=(16, 16))
    with sns.plotting_context("notebook", font_scale=1.4):
            sns.heatmap(corr_matrix, center=0, fmt=.6, cmap="coolwarm")
    plt.rcParams.update({'font.size': 16})
    plt.savefig(
        os.path.join(output_path, f"correlation_heatmap.{output_format}"), bbox_inches="tight"
    )


def perfomance_vs_characteristic_plot(
    results_df: pd.DataFrame,
    characteristics_df: pd.DataFrame,
    target_characteristic: str,
    characteristic_type: str,
    output_path: str,
    mode: str = "avg",
):
    data = prepare_data(results_df, characteristics_df, mode)
    data[target_characteristic] = data[target_characteristic].apply(lambda x: COLS_TO_KEEP_GLOBAL_CORRELATION[x] if x in COLS_TO_KEEP_GLOBAL_CORRELATION else x)
    
    sns.set(style="whitegrid")
    sns.set_palette("Set2")
    plt.figure(figsize=(10, 8))
    charac_display_name = CHARACTERISTICS_DISPLAY_NAMES[target_characteristic]
    plt.title(f"Model perfromance vs {charac_display_name.lower()}", fontsize=15)
    plt.xlabel(charac_display_name, fontsize=15)
    plt.ylabel("Average performance", fontsize=15)
    if characteristic_type == "categorical":
        sns.boxplot(data=data, x=target_characteristic, y="score")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    elif characteristic_type == "numerical":
        sns.scatterplot(data=data, x=target_characteristic, y="score")
        plt.xscale("log")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    else:
        raise ValueError(f"Unknown characteristic type: {characteristic_type}")

    plt.savefig(output_path, bbox_inches="tight")


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

    characteristics_df = pd.read_csv(args.characteristics_csv)
    characteristics_df["model"] = characteristics_df["model"].apply(
        lambda x: os.path.basename(x)
    )
    global_correlation(results_df, characteristics_df, args.output_folder, args.output_format)
    for k, v in CHARACTERISTICS.items():
        output_path = os.path.join(args.output_folder, f"perf_vs_{k}_avg.{args.output_format}")
        perfomance_vs_characteristic_plot(
            results_df, characteristics_df, k, v, output_path, mode="avg"
        )
