#!/usr/bin/env python


TASK_LIST = [
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
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "AskUbuntuDupQuestions",
    "SciDocs"
    "StackOverflowDupQuestions",
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
        all_results = {**all_results, **results}

csv_file = f"{results_folder}_results.csv"
print(f"Converting {results_folder} to {csv_file}")

with io.open(csv_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "metric", "value"])

    for task_name in TASK_LIST:
        writer.writerow([task_name, all_results["acc"], v["acc_stderr"], versions[k]])


    for k,v in sorted(results["results"].items()):
        if k not in versions:
            versions[k] = -1

        if "acc" in v:
            writer.writerow([k, "acc", v["acc"], v["acc_stderr"], versions[k]])
        if "acc_norm" in v:
            writer.writerow([k, "acc_norm", v["acc_norm"], v["acc_norm_stderr"], versions[k]])
        if "f1" in v:
            writer.writerow([k, "f1", v["f1"], v["f1_stderr"] if "f1_stderr" in v else "", versions[k]])
        # if "ppl" in v:
        #     writer.writerow([k, "ppl", v["ppl"], v["ppl_stderr"], versions[k]])
        # if "em" in v:
        #     writer.writerow([k, "em", v["em"], v["em_stderr"] if "em_stderr" in v else "", versions[k]])
