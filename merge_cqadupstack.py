"""
Usage: python merge_cqadupstack.py results_folder_path
"""
import json

TASK_LIST_CQA = [
    "CQADupstackAndroid",
    "CQADupstackEnglish",
    "CQADupstackGaming",
    "CQADupstackGisRetrieval"
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
]

NOAVG_KEYS = [
    "evaluation_time",
    "version"
]

import os
import sys
import json
import io
import csv
import glob

results_folder = sys.argv[1]
files = glob.glob(f'{results_folder.strip("/")}/CQADupstack*.json')

print("Found CQADupstack files: ", files)

if len(files) == len(TASK_LIST_CQA):
    all_results = {}
    for file_name in files:
        with io.open(file_name, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for split, split_results in results.items():
                all_results.setdefault(split, {})
                for metric, score in split_results.items():
                    all_results.setdefault(metric, 0)
                    if metric not in NOAVG_KEYS:
                        score = all_results[metric] + score * 1/len(TASK_LIST_CQA)
                    all_results[metric] = score


