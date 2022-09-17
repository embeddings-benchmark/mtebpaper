"""
Usage: python results_to_heatmap.py results_folder_path
results_folder_path contains results of multiple models whose folders should be named after them
Source: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
"""
import io
import json
import os
import sys

from mteb import MTEB
import numpy as np
import pandas as pd

TASK_LIST_BITEXT = [
    "BUCC",
    "Tatoeba",
]

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

TASK_LIST = (
    TASK_LIST_BITEXT
    + TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)

TASK_LIST_EN = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)

TASK_LIST_NAMES = [
    ("Classification", TASK_LIST_CLASSIFICATION, ["en", "en-en"]),
    ("Clustering", TASK_LIST_CLUSTERING, ["en", "en-en"]),
    ("PairClassification", TASK_LIST_PAIR_CLASSIFICATION, ["en", "en-en"]),
    ("Reranking", TASK_LIST_RERANKING, ["en", "en-en"]),
    ("Retrieval", TASK_LIST_RETRIEVAL, ["en", "en-en"]),
    ("STS", TASK_LIST_STS, ["en", "en-en"]),
    ("all", TASK_LIST, ["en", "en-en"]),
    ("BitextMining", TASK_LIST_BITEXT, []),
]

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
    rows = []
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


MODELS_TO_CORRELATE = [
    ("Glove", "glove.6B.300d"),
    ("Komninos", "komninos"),
    ("LASER2", "LASER2"),
    # ("LaBSE", "LaBSE"),
    ("bert-base-uncased", "bert-base-uncased"),
    ("BERT Co-Condensor", "msmarco-bert-co-condensor"),
    ("SimCSE-bert-base-unsup", "unsup-simcse-bert-base-uncased"),
    ("SimCSE-bert-base-sup", "sup-simcse-bert-base-uncased"),
    ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
    ("all-mpnet-base-v2", "all-mpnet-base-v2"),
    ("Contriever", "contriever-base-msmarco"),
    ("SGPT-125M-nli", "SGPT-125M-weightedmean-nli-bitfit",),
    ("SGPT-125M-msmarco", "SGPT-125M-weightedmean-msmarco-specb-bitfit",),
    ("SGPT-5.8B-nli", "SGPT-5.8B-weightedmean-nli-bitfit",),
    ("SGPT-5.8B-msmarco", "SGPT-5.8B-weightedmean-msmarco-specb-bitfit",),
    ("GTR-Base", "gtr-t5-base",), # 110M
    ("GTR-XXL", "gtr-t5-xxl",), # 4.8B
    ("ST5-Base", "sentence-t5-base",), # 110M
    ("ST5-XXL", "sentence-t5-xxl",), # 4.8B
]

### MODEL HEATMAP

model_dict = []

for ds in TASK_LIST_EN:
    model_dict.append({model[0]: get_row(ds, model[-1], limit_langs=["en", "en-en"]) for model in MODELS_TO_CORRELATE})

model_df = pd.DataFrame(model_dict)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(model_df.corr(), dtype=np.bool))
heatmap = sns.heatmap(model_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Pearson Correlations of scores on MTEB', fontdict={'fontsize':18}, pad=16);

plt.savefig('heatmap_model.png', dpi=300, bbox_inches='tight')

data_dict = []


### DATA HEATMAP
# This is to be differentiated from a heatmap of actual data content (e.g. via unigram Jaccard similarity)
# E.g. for BEIR SciFact & HotpotQA have very low unigram Jaccard similarity, but in this method,
# they get a high similarity score, because model scores seem to correlate on the datasrt

for model in MODELS_TO_CORRELATE:
    data_dict.append({ds: get_row(ds, model[-1], limit_langs=["en", "en-en"]) for ds in TASK_LIST_EN})

data_df = pd.DataFrame(data_dict)

plt.figure(figsize=(128, 48))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(data_df.corr(), dtype=np.bool))
heatmap = sns.heatmap(data_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Pearson Correlations of scores on MTEB', fontdict={'fontsize':18}, pad=16);

plt.savefig('heatmap_data.png', dpi=300, bbox_inches='tight')
