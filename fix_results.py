"""
Usage: python fix_results.py results_folder_path
"""
import json
import os
import sys
import json
import io
import csv
import glob

results_folder = sys.argv[1]
files = glob.glob(f'{results_folder.strip("/")}/*.json')

print("Found files: ", files)

for file_name in files:
    with io.open(file_name, 'r', encoding='utf-8') as f:
        results = json.load(f)
        if "dataset_version" not in results:
            results["dataset_version"] = None
        if "mteb_version" not in results:
            results["mteb_version"] = "0.0.2"
        if "STS22" in file_name:
            for split, split_results in results.items():
                if split not in ("train", "validation", "test"):
                    continue
                for metric, score in split_results.items():
                    results[split][metric] = abs(score)
                results.setdefault(split, {})
        with io.open(file_name, 'w', encoding='utf-8') as f:
            json.dump(results, f)

