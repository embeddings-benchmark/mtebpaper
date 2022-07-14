"""
Usage: python results_to_csv results_folder_path
"""

from mteb import MTEB

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
    "SciDocs",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
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
    "SummEval",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS

import os
import sys
import json
import io
import csv

results_folder = sys.argv[1]
all_results = {}

for file_name in os.listdir(results_folder):
    if not file_name.endswith(".json"):
        print(f"Skipping non-json {file_name}")
        continue
    with io.open(os.path.join(results_folder, file_name), 'r', encoding='utf-8') as f:
        results = json.load(f)
        all_results = {**all_results, **{file_name.replace(".json", ""): results}}

csv_file = f"{results_folder}_results.csv"
print(f"Converting {results_folder} to {csv_file}")

with io.open(csv_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "metric", "value"])

    for task_name in TASK_LIST:
        tasks = MTEB(tasks=[task_name]).tasks
        if len(tasks) == 0:
            print(f"No tasks found for name {task_name}")
        main_metric = tasks[0].description["main_score"]
        test_result = all_results.get(task_name, {}).get("test")
        if test_result is None:
            print(f"{task_name} / test set not found - Writing empty row")
            writer.writerow([task_name, main_metric, ""])
            continue
        if "en" in test_result:
            test_result = test_result["en"]
        elif "en-en" in test_result:
            test_result = test_result["en-en"]
        if main_metric == "cosine_spearman":
            test_result = test_result.get("cos_sim", {}).get("spearman")
        elif main_metric == "ap":
            test_result = test_result.get("cos_sim", {}).get("ap")
        else:
            test_result = test_result.get(main_metric)
        if test_result is None:
            print(f"{main_metric} not found for task {task_name}")
            writer.writerow([task_name, main_metric, ""])
            continue
        writer.writerow([task_name, main_metric, test_result])
