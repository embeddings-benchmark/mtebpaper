"""
Usage: python results_to_csv.py results_folder_path
Make sure the final directory results_folder_path is the name of your model
"""
import csv
import io
import json
import os
import sys

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
    "SummEval",
]

TASK_LIST_BITEXT = [
    "BUCC",
    "Tatoeba",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS

results_folder = sys.argv[1]
results_folder = results_folder.strip("/")
model_name = results_folder.split("/")[-1]
print(f"Using model name {model_name}")

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

NOT_FOUND = []

def get_rows(task_name, limit_langs=[]):
    rows = []
    # CQADupstackRetrieval uses the same metric as its subsets
    tasks = MTEB(tasks=[task_name.replace("CQADupstackRetrieval", "CQADupstackTexRetrieval")]).tasks
    assert len(tasks) == 1, f"Found {len(tasks)} for {task_name}. Expected 1."
    main_metric = tasks[0].description["main_score"]
    test_result = all_results.get(task_name, {})
    
    # Dev / Val set is used for MSMARCO (See BEIR paper)
    if "MSMARCO" in task_name:
        test_result = test_result.get("dev") if "dev" in test_result else test_result.get("validation")
    else:
        test_result = test_result.get("test")
    if test_result is None:
        print(f"{task_name} - test set not found")
        NOT_FOUND.append(task_name)
        return [[model_name, task_name, "", main_metric, ""]]

    for lang in tasks[0].description["eval_langs"]:
        if limit_langs and lang not in limit_langs: continue
        test_result_lang = test_result.get(lang, test_result)
        if main_metric == "cosine_spearman":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("spearman")
        elif main_metric == "ap":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("ap")
        else:
            test_result_lang = test_result_lang.get(main_metric)

        if test_result_lang is None:
            print(f"{lang} & {main_metric} not found for task {task_name}.")
            rows.append([model_name, task_name, lang, main_metric, ""])
        rows.append([model_name, task_name, lang, main_metric, test_result_lang])
    return rows

with io.open(csv_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["model", "dataset", "language", "metric", "value"])
    for task_name in TASK_LIST:
        writer.writerows(get_rows(task_name))

    # Add average scores
    for shortcut, task_list in [("clf",TASK_LIST_CLASSIFICATION), ("cls", TASK_LIST_CLUSTERING), ("pairclf", TASK_LIST_PAIR_CLASSIFICATION), ("rrk", TASK_LIST_RERANKING), ('rtr', TASK_LIST_RETRIEVAL), ('sts', TASK_LIST_STS), ("all", TASK_LIST)]:
        if all([x in all_results for x in task_list]):
            rows = [y for x in task_list for y in get_rows(x, limit_langs=["en", "en-en"])]
            avg = sum([float(x[-1]) for x in rows]) / len(rows)
            metric = "multiple" if shortcut == "all" else rows[-1][-2]
            writer.writerow([model_name, "en", f"average_{shortcut}", metric, avg])


print("Not found: " + "'"+"','".join(NOT_FOUND)+"'", len(NOT_FOUND))
