"""
Usage: 
Inspired by Fig 3 from https://arxiv.org/pdf/2011.04006.pdf
"""
import io
import json
import os
import sys

import matplotlib.pyplot as plt
from mteb import MTEB


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


TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST_EN = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)

MODEL_TO_NAME = {
    "gtr-t5-base": "GTR-Base",
    "gtr-t5-large": "GTR-Large",
    "gtr-t5-xl": "GTR-XL",
    "gtr-t5-xxl": "GTR-XXL",
    "sentence-t5-base": "ST5-Base",
    "sentence-t5-large": "ST5-Large",
    "sentence-t5-xl": "ST5-XL",
    "sentence-t5-xxl": "ST5-XXL",
    "SGPT-125M-weightedmean-msmarco-specb-bitfit": "SGPT-125M-msmarco",
    "SGPT-1.3B-weightedmean-msmarco-specb-bitfit": "SGPT-1.3B-msmarco",
    "SGPT-2.7B-weightedmean-msmarco-specb-bitfit": "SGPT-2.7B-msmarco",
    "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": "SGPT-5.8B-msmarco",
    "sgpt-bloom-7b1-msmarco": "SGPT-7.1B-msmarco",
    "SGPT-125M-weightedmean-nli-bitfit": "SGPT-125M-nli",
    "SGPT-5.8B-weightedmean-nli-bitfit": "SGPT-5.8B-nli",
    "sup-simcse-bert-base-uncased": "SimCSE-bert-base-sup",
    "contriever-base-msmarco": "Contriever",
    "msmarco-bert-co-condensor": "BERT Co-Condensor",
    "unsup-simcse-bert-base-uncased": "SimCSE-bert-base-unsup",
    "glove.6B.300d": "Glove",
    "komninos": "Komninos",
    "all-MiniLM-L6-v2": "MiniLM-L6-v2",
    "all-mpnet-base-v2": "MPNet-base-v2",
}

NAME_TO_ARCH = {
    "gtr": ("T5", "darkorange"),
    "st5": ("T5", "darkorange"),
    "sgpt": ("GPT", "deepskyblue"),
    "simcse": ("BERT", "gold"),
    "contriever": ("BERT", "gold"),
    "bert": ("BERT", "gold"),
    "mpnet": ("MPNet", "steelblue"),
    "minilm": ("MiniLM", "lightblue"),
    "laser2": ("LASER", "darkblue"),
    "labse": ("BERT", "gold"),
    "glove": ("WordEmbeddings", "blue"),
    "komninos": ("WordEmbeddings", "blue"),
}


### LOGIC ###

# Get average MTEB performance

results_folder = sys.argv[1].strip("/")
benchmark_json = sys.argv[2]

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

results_avg = {}

for model in all_results:
    try:
        model_task_results = [get_row(task, model, limit_langs=["en", "en-en"]) for task in TASK_LIST_EN]
    except:
        continue
    results_avg[model] = 100 * (sum(model_task_results) / len(model_task_results))


with open(benchmark_json, "r") as f:
    gpu_bench = json.load(f)

import numpy as np

fig, ax = plt.subplots(figsize=(8,8))

for k, v in gpu_bench.items():
    if k == "specs":
        continue
    model_name = MODEL_TO_NAME.get(k, k)
    (model_arch, color) = NAME_TO_ARCH.get(model_name.split(" ")[0].split("-")[0].lower(), (model_name, None))

    if k not in results_avg:
        print(f"Missing average score for {k}")
        continue

    speed = 1000 / v["STS15"]["speed_ms"]
    score = results_avg[k]
    
    ax.scatter(
        speed, 
        score,
        label=model_arch,
        color=color,
        s=v["STS15"]["embedding_size_kb"] * 150, 
        alpha=.5
    )
    # Empirical offsets
    x_offset = y_offset = 0
    if model_name in ("ST5-Base"):
        x_offset = -0.5 * speed
    elif model_name in ("GTR-Base"):
        x_offset = 0.7 * speed
        y_offset = -0.01 * score
    elif model_name in ("Contriever"):
        x_offset = 0.75 * speed
    elif model_name in ("LaBSE"):
        x_offset = 0.65 * speed
        y_offset = 0.01 * score
    elif model_name in ("GTR-XXL", "ST5-XXL"):
        x_offset = -1.1 * speed
        if model_name == "GTR-XXL":
            y_offset = 0.01 * score
    elif model_name == "Komninos":
        x_offset = 0.2 * speed
        y_offset = 0.03 * score
    elif model_name == "Glove":
        x_offset = 0.2 * speed
        y_offset = -0.025 * score
    elif model_name.startswith("SGPT-5.8B"):
        x_offset = 0.3 * speed
        y_offset = 0.04 * score
    elif model_arch in ("BERT", "MiniLM", "MPNet", "LASER") or model_name.startswith("SGPT-125M"):
        x_offset = -0.5 * speed
    
    ax.text(
        speed - x_offset,
        score - y_offset,
        model_name,
    )

    # Annotate does not work with logscale, https://stackoverflow.com/questions/21140385/matplotlib-annotate-doesnt-work-on-log-scale
    #ax.annotate(
    #    MODEL_TO_NAME.get(k, k), 
    #   xy=(np.log10(1000 / v["STS15"]["speed_ms"]), results_avg[k] - offset)
    #)

ax.set_xlabel("Speed (examples per sec)")
ax.set_ylabel("MTEB Score")
ax.set_xscale('log')
ax.grid(alpha=0.5)

# Create deduplicated Global Legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
lgnd = plt.legend(
    by_label.values(), 
    by_label.keys(), 
    title="Base Architecture",
    loc=(0.08,0.08), # "lower left", 
)
# Rescale bubbles to have the same size
for handle in lgnd.legendHandles:
    handle.set_sizes([70.0])
    


plt.savefig('benchmark.png', dpi=300, bbox_inches='tight')
