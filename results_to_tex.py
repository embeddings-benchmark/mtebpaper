"""
Usage: python results_to_csv.py results_folder_path
results_folder_path contains results of multiple models whose folders should be named after them
"""
from curses.ascii import TAB
import io
import json
import os
import sys

from mteb import MTEB
import numpy as np

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


def get_rows(dataset, model_name, limit_langs=[], skip_langs=[]):
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
            rows.append([lang, main_metric, None])
            continue

        test_result_lang = test_result.get(lang, test_result)
        if main_metric == "cosine_spearman":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("spearman")
        elif main_metric == "ap":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("ap")
        else:
            test_result_lang = test_result_lang.get(main_metric)

        if test_result_lang is None:
            rows.append([lang, main_metric, None])
            continue

        rows.append([lang, main_metric, test_result_lang])
    return rows


# Models to include in CLF Table, (appearance_name, model_name)
CLF_MODELS = [
    ("bert-base-uncased", "bert-base-uncased"),
    ("all-mpnet-base-v2", "all-mpnet-base-v2"),
    ("SGPT-5.8B-msmarco", "SGPT-5.8B-weightedmean-msmarco-specb-bitfit",),
    ("GTR-XXL", "gtr-t5-xxl",),
    ("ST5-XXL", "sentence-t5-xxl",),
]

TABLE = "Dataset & " + " & ".join([x[0] for x in CLF_MODELS]) + " \\\\" + "\n"

scores_all = []
for ds in TASK_LIST_CLASSIFICATION:
    results =  [x for model in CLF_MODELS for x in get_rows(dataset=ds, model_name=model[-1], limit_langs=["en"])]
    scores = [x[-1] for x in results]
    scores_all.append(scores)
    one_line = " & ".join([ds] + [str(round(x*100, 2)) for x in scores])
    TABLE += one_line + " \\\\" + "\n"

scores_avg =  list(np.sum(np.array(scores_all), axis=0) / len(TASK_LIST_CLASSIFICATION))

TABLE += " & ".join(["average"] + [str(round(x*100, 2)) for x in scores_avg]) + " \\\\" + "\n"



BITEXT_MODELS = MULTILING_MODELS = [
    ("Glove", "glove.6B.300d"),
    ("Komninos", "komninos"),
    ("LASER2", "LASER2"),
    ("LaBSE", "LaBSE"),
    ("paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-multilingual-MiniLM-L12-v2"),
    ("paraphrase-multilingual-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2"),
    ("SGPT-bloom-7b1-msmarco", "sgpt-bloom-7b1-msmarco"),
    ("SGPT-bloom-1b3-nli", "sgpt-bloom-1b3-nli"),
]

UNSUPERVISED_MODELS = [
    ("Glove", "glove.6B.300d"),
    ("Komninos", "komninos"),
    ("LASER2", "LASER2"),
    ("LaBSE", "LaBSE"),
    ("bert-base-uncased", "bert-base-uncased"),
    ("BERT Co-Condensor", "msmarco-bert-co-condensor"),
    ("SimCSE-bert-base-unsup", "unsup-simcse-bert-base-uncased"),
]

SUPERVISED_MODELS = [
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



def get_table(models, task_list, limit_langs=[], skip_langs=[], name="table"):
    TABLE = "Dataset & Language & " + " & ".join([x[0] for x in models]) + " \\\\" + "\n"
    scores_all = [] # TODO: Average
    for ds in task_list:
        results =  [get_rows(dataset=ds, model_name=model[-1], limit_langs=limit_langs, skip_langs=skip_langs) for model in models]
        assert all(len(sub) == len(results[0]) for sub in results)
        for lang_idx in range(len(results[0])):
            scores = [x[lang_idx][-1] for x in results]
            scores_all.append(scores)
            lang = results[0][lang_idx][0]
            one_line = " & ".join([ds, lang] + [str(round(x*100, 2)) if x else "" for x in scores])
            TABLE += one_line + " \\\\" + "\n"

    arr = np.array(scores_all, dtype=np.float32)
    # Get an index of columns which has any NaN value
    index = np.isnan(arr).any(axis=0)
    # Delete columns (models) with any NaN value from 2D NumPy Array
    arr = np.delete(arr, index, axis=1)
    # Average
    scores_avg = list(np.mean(arr, axis=0))
    # Insert empty string for NaN columns
    for i, val in enumerate(index):
        if val == True:
            scores_avg.insert(i, "")
    lang = "mix" if not(limit_langs) else limit_langs[0]
    TABLE += " & ".join(["Average", lang] + [str(round(x*100, 2)) if x else "" for x in scores_avg]) + " \\\\" + "\n"

    with open(f"{name}.txt", "w") as f:
        f.write(TABLE)


get_table(UNSUPERVISED_MODELS, TASK_LIST_EN, limit_langs=["en", "en-en",], name="unsupervised_en")
get_table(SUPERVISED_MODELS, TASK_LIST_EN, limit_langs=["en", "en-en",], name="supervised_en")
get_table(BITEXT_MODELS, TASK_LIST_BITEXT, limit_langs=[], name="bitext")
get_table(MULTILING_MODELS, TASK_LIST_CLASSIFICATION, limit_langs=[], skip_langs=["en", "en-en", "en-ext"], name="multilingclf")
get_table(MULTILING_MODELS, TASK_LIST_STS, limit_langs=[], skip_langs=["en", "en-en", "en-ext"], name="multilingsts")
    
