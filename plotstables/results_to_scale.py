"""
Creates scaling graphs
Usage: python results_to_scale.py results_folder_path
results_folder_path contains results of multiple models whose folders should be named after them
"""
import io
import json
import os
import sys

from mteb import MTEB
import numpy as np

### GLOBAL VARIABLES ###

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]

# Parameter counts in millions
MODELS = [
# Doesnt add a lot of value to the figure
#    [
#        ("MiniLM-L6", "all-MiniLM-L6-v2", 22.713216), # 22.7 M
#        ("MiniLM-L12", "all-MiniLM-L12-v2", 33.360000), # 33.4 M
#    ],
    [
        ("GTR-Base", "gtr-t5-base", 110),
        ("GTR-Large", "gtr-t5-large", 335),
        ("GTR-XL", "gtr-t5-xl", 1240),
        ("GTR-XXL", "gtr-t5-xxl", 4800),
    ],
    [
        ("ST5-Base", "sentence-t5-base", 110),
        ("ST5-Large", "sentence-t5-large", 335),
        ("ST5-XL", "sentence-t5-xl", 1240),
        ("ST5-XXL", "sentence-t5-xxl", 4800),
    ],
    [
        ("SGPT-125M-msmarco", "SGPT-125M-weightedmean-msmarco-specb-bitfit", 125),
        ("SGPT-1.3B-msmarco", "SGPT-1.3B-weightedmean-msmarco-specb-bitfit", 1300),
        ("SGPT-2.7B-msmarco", "SGPT-2.7B-weightedmean-msmarco-specb-bitfit", 2700),
        ("SGPT-5.8B-msmarco", "SGPT-5.8B-weightedmean-msmarco-specb-bitfit", 5800),
    ],
]

# todo: remove
lines = ["blue2", "blue", "purple"]
shades = ["lightblue2", "lightblue", "lightpurple"]
colors = {
    "purple": "#7B3FB9",
    "lightpurple": "#CBB3E3",
    "blue": "#221D91",
    "lightblue": "#B6B4DB",
    "blue2": "#86D4F1",
    "lightblue2": "#AAF2F2",
}


MODEL_TO_MARKER = {
    "MiniLM": "o",
    "GTR": "x",
    "ST5": "*",
    "SGPT": "v",
}

# Base from:
# https://coolors.co/palette/ff5400-ff6d00-ff8500-ff9100-ff9e00-00b4d8-0096c7-0077b6-023e8a-03045e
# Yellow tones from:
# https://coolors.co/palette/6ab6dc-49a6d4-2f94c6-277ba5-1f6284-e0b700-ffd20a-ffda33-ffe15c-ffe570
# Green from:
# https://coolors.co/palette/f94144-f3722c-f8961e-f9844a-f9c74f-90be6d-43aa8b-4d908e-577590-277da1
MODEL_TO_COLOR = {
    "MiniLM": "#BAF19C",#"#017600", # Green
    "MPNet": "#F94144",#"#007A7A", # Light Green
    "GTR": "#FF5400",#"#221D91", # Blue 1
    "ST5": "#FF9E00",#"#86D4F1", # Blue 2
    "SGPT": "#00B4D8",#"#7B3FB9", # Purple
    "SimCSE": "#F9C74F",#"#2070B4", # Blue 3
    "LaBSE": "#F9C74F",#"#2070B4", # Blue 3
    "SPECTER": "#E0B700", # Shade of #2070B4
    "Glove": "#023E8A",#"#9BC7DD", # Light Blue
    "LASER2": "#03045E", # Grey
}


TASK_LIST_NAMES = [
    ("Classification", TASK_LIST_CLASSIFICATION, ["en", "en-en"], "accuracy"),
    ("Clustering", TASK_LIST_CLUSTERING, ["en", "en-en"], "v_measure"),
    ("PairClassification", TASK_LIST_PAIR_CLASSIFICATION, ["en", "en-en"], "ap"),
    ("Reranking", TASK_LIST_RERANKING, ["en", "en-en"], "map"),
    ("Retrieval", TASK_LIST_RETRIEVAL, ["en", "en-en"], "nDCG@10"),
    ("STS", TASK_LIST_STS, ["en", "en-en"], "cos. sim. spearman corr."),
]


### LOGIC ###

results_folder = sys.argv[1].strip("/")

all_results = {}

for model_name in os.listdir(results_folder):
    model_res_folder = os.path.join(results_folder, model_name)
    if os.path.isdir(model_res_folder):
        all_results.setdefault(model_name, {})
        for file_name in os.listdir(model_res_folder):
            if not file_name.endswith(".json"):
                print(f"Skipping non-json {file_name}")
                continue
            with io.open(os.path.join(model_res_folder, file_name), "r", encoding="utf-8") as f:
                results = json.load(f)
                all_results[model_name] = {**all_results[model_name], **{file_name.replace(".json", ""): results}}


def get_row(dataset, model_name, limit_langs=[], skip_langs=[]):
    # CQADupstackRetrieval uses the same metric as its subsets
    tasks = MTEB(tasks=[dataset.replace("CQADupstackRetrieval", "CQADupstackTexRetrieval")]).tasks
    assert len(tasks) == 1, f"Found {len(tasks)} for {dataset}. Expected 1."
    main_metric = tasks[0].description["main_score"]
    test_result = all_results.get(model_name, {}). get(dataset, {})

    # Dev / Val set is used for MSMARCO (See BEIR paper)
    if "MSMARCO" in dataset:
        test_result = (
            test_result.get("dev") if "dev" in test_result else test_result.get("validation")
        )
    else:
        test_result = test_result.get("test")

    for lang in tasks[0].description["eval_langs"]:
        if (limit_langs and lang not in limit_langs) or (skip_langs and lang in skip_langs):
            continue
        elif test_result is None:
            raise NotImplementedError(f"Got no test result {test_result} for ds: {dataset} model: {model_name}")

        test_result_lang = test_result.get(lang, test_result)
        if main_metric == "cosine_spearman":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("spearman")
        elif main_metric == "ap":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("ap")
        else:
            test_result_lang = test_result_lang.get(main_metric)

        if test_result_lang is None:
            raise NotImplementedError

        return test_result_lang
    raise NotImplementedError


# Create a plot for each task with scaling of the model performances on this task
import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(16, 20), facecolor='w', edgecolor='k', ncols=2, nrows=3, sharey=False)


# Create each boxplot
model_xticks_global = ['0.1B', '1B','2B','4B']
model_xticks_num_global = [np.log10(100_000_000), np.log10(1_000_000_000), np.log10(2_000_000_000), np.log10(4_000_000_000)]

for ax, (task_name, task_list, limit_langs, metric) in zip(axes.flatten(), TASK_LIST_NAMES):
    for i, model_group in enumerate(MODELS):
        model_xticks_num = [np.log10(x[-1] * 1_000_000) for x in model_group]
        avg_scores = []
        std_scores = []
        for model in model_group:
            model_name = model[0]
            try:
                model_task_results = [get_row(task, model[1], limit_langs=limit_langs) for task in task_list]
            except:
                model_task_results = [0.5]
            
            avg_scores.append(np.mean(np.array(model_task_results)).item())
            std_scores.append(np.std(np.array(model_task_results)).item())

        ax.plot(
            model_xticks_num, 
            avg_scores, 
            label=model_name.split("-")[0], 
            color=MODEL_TO_COLOR.get(model_name.split("-")[0]),
            marker=MODEL_TO_MARKER.get(model_name.split("-")[0])
        )

        # Shade doesn't look good, as std is too big
        # ax.fill_between(model_xticks_num, [avg-std for avg, std in zip(avg_scores, std_scores)], [avg+std for avg, std in zip(avg_scores, std_scores)], color=colors.get(shades[i]), alpha=0.5)

    ax.set_ylabel(f"Average Performance ({metric})", fontsize=15)
    ax.set_xlabel("Model Parameters (Billions)", fontsize=15)
    ax.set_xticks(model_xticks_num_global, model_xticks_global)
    ax.set_title(task_name, fontweight="bold", fontsize=20)
    ax.grid(alpha=0.5)

# Create deduplicated Global Legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    by_label.values(), 
    by_label.keys(), 
    loc=(0.35, 0.94), # "upper center",
    ncol=len(by_label),
    frameon=False,
    fontsize=15,
)

plt.savefig('scale.png', dpi=300, bbox_inches='tight')
