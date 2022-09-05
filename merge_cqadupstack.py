"""
Merges CQADupstack subset results
Usage: python merge_cqadupstack.py results_folder_path
"""
import json

TASK_LIST_CQA = [
    "CQADupstackAndroid",
    "CQADupstackEnglish",
    "CQADupstackGaming",
    "CQADupstackGisRetrieval",
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
    "evaluation_time", # TODO: Sum this one
    "version"
]

import os
import sys
import json
import io
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
                if split not in ("train", "validation", "test"):
                    all_results[split] = split_results
                    continue
                all_results.setdefault(split, {})
                for metric, score in split_results.items():
                    all_results[split].setdefault(metric, 0)
                    if metric not in NOAVG_KEYS:
                        score = all_results[split][metric] + score * 1/len(TASK_LIST_CQA)
                    all_results[split][metric] = score

    print("Saving ", all_results)
    with io.open(os.path.join(results_folder, "CQADupstackRetrieval.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f)
else:
    print(f"Missing files {set(TASK_LIST_CQA) - set(files)} or got too many files.")
