"""
Usage: python results_to_multilingual.py results_folder_path
Make sure the final directory results_folder_path is the name of your model
"""
import json
import os
import sys

### GLOBAL VARIABLES ###

TASK_LIST_BITEXT = [
    "BUCC",
    "Tatoeba",
]

BITEXT_MODELS = MULTILING_MODELS = [
    "LaBSE",
    "LASER2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "sgpt-bloom-7b1-msmarco",
    # "sgpt-bloom-1b3-nli", # Not too interesting
]

MODEL_TO_NAME = {
    "bert-base-uncased": "BERT",
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
    "sgpt-bloom-7b1-msmarco": "SGPT-BLOOM-7.1B-msmarco",
    "SGPT-125M-weightedmean-nli-bitfit": "SGPT-125M-nli",
    "SGPT-5.8B-weightedmean-nli-bitfit": "SGPT-5.8B-nli",
    "sup-simcse-bert-base-uncased": "SimCSE-BERT-sup",
    "contriever-base-msmarco": "Contriever",
    "msmarco-bert-co-condensor": "coCondenser-msmarco", # They write it as coCondenser in the paper
    "unsup-simcse-bert-base-uncased": "SimCSE-BERT-unsup",
    "glove.6B.300d": "Glove",
    "komninos": "Komninos",
    "all-MiniLM-L6-v2": "MiniLM-L6",
    "all-MiniLM-L12-v2": "MiniLM-L12",
    "paraphrase-multilingual-MiniLM-L12-v2": "MiniLM-L12-multilingual",
    "all-mpnet-base-v2": "MPNet",
    "paraphrase-multilingual-mpnet-base-v2": "MPNet-multilingual",
    "allenai-specter": "SPECTER",
    "text-similarity-ada-001": "Ada Similarity",
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


MULTILINGUAL_CLF = [
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
]

MULTILINGUAL_STS = [
    "STS17",
    "STS22",
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
            with open(os.path.join(model_res_folder, file_name), "r", encoding="utf-8") as f:
                results = json.load(f)
                all_results[model_name] = {**all_results[model_name], **{file_name.replace(".json", ""): results}}



# Create a plot for each task with scaling of the model performances on this task
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(64,12))

markers = ["x", "o", "v", "*", "p"]

# Compute averages
scores = {}
for i, model in enumerate(BITEXT_MODELS):
    if not(all_results.get(model, []).get("Tatoeba")):
        continue
    for lang, res in all_results[model]["Tatoeba"]["test"].items():
        if lang == "evaluation_time":
            continue
        scores.setdefault(lang, [])
        scores[lang].append(res["f1"])
# Average
scores = {k: np.mean(v) for k,v in scores.items()}
scores_sorted = sorted(scores.items(), key=lambda x: x[-1], reverse=True)
langs_sorted = [x[0] for x in scores_sorted]
global_idx = {lang: langs_sorted.index(lang) for lang in scores}

for i, model in enumerate(BITEXT_MODELS):
    scores = {}

    if not(all_results.get(model, []).get("Tatoeba")):
        continue

    for lang, res in all_results[model]["Tatoeba"]["test"].items():
        if lang == "evaluation_time":
            continue
        scores[lang] = res["f1"]
    
    # Optionally sort by LaBSE scores
    if i == 0: 
        assert model == "LaBSE"
        scores_sorted = sorted(scores.items(), key=lambda x: x[-1], reverse=True)
        langs_sorted = [x[0] for x in scores_sorted]
        global_idx = {lang: langs_sorted.index(lang) for lang in scores}

    # Reverse is already accounted for in global_idx
    scores_sorted = sorted(scores.items(), key=lambda x: global_idx[x[0]], reverse=False)
    x_langs = [x[0] for x in scores_sorted]
    y_scores = [x[1] for x in scores_sorted]

    model_name = MODEL_TO_NAME.get(model, model)
    ax.plot(x_langs, y_scores, linewidth=6.0, label=model_name, marker=markers[i], color=MODEL_TO_COLOR.get(model_name.split("-")[0]))

ax.set_ylabel("F1 score", fontsize=20)
ax.margins(x=0.01) # Reduce whitespace left & right

plt.xticks(rotation=45) #plt.xticks(rotation=90, ha='right')
plt.legend(fontsize=25)
plt.savefig('multilingual_tatoeba.png', dpi=300, bbox_inches='tight')


### CLASSIFICATION ###

# Compute averages
scores = {}
for i, model in enumerate(BITEXT_MODELS):
    for ds in MULTILINGUAL_CLF:
        if not(all_results.get(model, []).get(ds)):
            continue
        for lang, res in all_results[model][ds]["test"].items():
            if lang == "evaluation_time":
                continue
            elif lang == "en-ext":
                lang = "en"
            
            scores.setdefault(lang, [])
            scores[lang].append(res["accuracy"])
# Average
scores = {k: np.mean(v) for k,v in scores.items()}
scores_sorted = sorted(scores.items(), key=lambda x: x[-1], reverse=True)
langs_sorted = [x[0] for x in scores_sorted]
global_idx = {lang: langs_sorted.index(lang) for lang in scores}


fig, ax = plt.subplots(figsize=(32,8))


for i, model in enumerate(BITEXT_MODELS):
    scores = {}
    for ds in MULTILINGUAL_CLF:
        if not(all_results.get(model, []).get(ds)):
            continue
        for lang, res in all_results[model][ds]["test"].items():
            if lang == "evaluation_time":
                continue
            elif lang == "en-ext":
                lang = "en"
            
            scores.setdefault(lang, [])
            scores[lang].append(res["accuracy"])

    # Average scores for langs
    scores = {k: np.mean(v) for k,v in scores.items()}

    # Reverse is already accounted for in global_idx
    scores_sorted = sorted(scores.items(), key=lambda x: global_idx[x[0]], reverse=False)
    x_langs = [x[0] for x in scores_sorted]
    y_scores = [x[1] for x in scores_sorted]
    model_name = MODEL_TO_NAME.get(model, model)
    ax.plot(x_langs, y_scores, linewidth=6.0, label=model_name, marker=markers[i], color=MODEL_TO_COLOR.get(model_name.split("-")[0]))

ax.set_ylabel("Accuracy", fontsize=20)

plt.xticks(rotation=45) #plt.xticks(rotation=90, ha='right')
# plt.legend(fontsize=15)

plt.savefig('multilingual_clf.png', dpi=300, bbox_inches='tight')



### STS ###

# Compute averages
scores_multi = {}
scores_cross = {}

for i, model in enumerate(BITEXT_MODELS):
    for ds in MULTILINGUAL_STS:
        if not(all_results.get(model, []).get(ds)):
            continue
        for lang, res in all_results[model][ds]["test"].items():
            if lang == "evaluation_time":
                continue
            multi = True
            if "-" in lang:
                l1, l2 = lang.split("-")
                if l1 != l2:
                    multi = False
                else:
                    lang = l1
            if multi:
                scores_multi.setdefault(lang, [])
                scores_multi[lang].append(res["cos_sim"]["spearman"])
            else:
                scores_cross.setdefault(lang, [])
                scores_cross[lang].append(res["cos_sim"]["spearman"])

# Average
scores = {k: np.mean(v) for k,v in scores_multi.items()}
scores_sorted = sorted(scores.items(), key=lambda x: x[-1], reverse=True)
langs_sorted = [x[0] for x in scores_sorted]
global_idx_multi = {lang: langs_sorted.index(lang) for lang in scores}

scores = {k: np.mean(v) for k,v in scores_cross.items()}
scores_sorted = sorted(scores.items(), key=lambda x: x[-1], reverse=True)
langs_sorted = [x[0] for x in scores_sorted]
global_idx_cross = {lang: langs_sorted.index(lang) for lang in scores}



fig, axes = plt.subplots(figsize=(32,8), ncols=2, nrows=1, sharey=True)

ax_multi, ax_cross = axes

for i, model in enumerate(BITEXT_MODELS):
    scores_multi = {}
    scores_cross = {}
    for ds in MULTILINGUAL_STS:
        if not(all_results.get(model, []).get(ds)):
            continue
        for lang, res in all_results[model][ds]["test"].items():
            if lang == "evaluation_time":
                continue
            multi = True
            if "-" in lang:
                l1, l2 = lang.split("-")
                if l1 != l2:
                    multi = False
                else:
                    lang = l1
            
            if multi:
                scores_multi.setdefault(lang, [])
                scores_multi[lang].append(res["cos_sim"]["spearman"])
            else:
                scores_cross.setdefault(lang, [])
                scores_cross[lang].append(res["cos_sim"]["spearman"])              

    scores_multi = {k: np.mean(v) for k,v in scores_multi.items()}
    scores_cross = {k: np.mean(v) for k,v in scores_cross.items()}

    scores_sorted_multi = sorted(scores_multi.items(), key=lambda x: global_idx_multi[x[0]], reverse=False)
    scores_sorted_cross = sorted(scores_cross.items(), key=lambda x: global_idx_cross[x[0]], reverse=False)

    model_name = MODEL_TO_NAME.get(model, model)
    model_color = MODEL_TO_COLOR.get(model_name.split("-")[0])

    ax_multi.plot(
        [x[0] for x in scores_sorted_multi],
        [x[1] for x in scores_sorted_multi],
        label=model_name, 
        marker=markers[i], 
        color=model_color,
        linewidth=6.0,
    )

    ax_cross.plot(        
        [x[0] for x in scores_sorted_cross],
        [x[1] for x in scores_sorted_cross],
        label=model_name,
        marker=markers[i], 
        color=model_color,
        linewidth=6.0,
    )

ax_multi.set_ylabel("Cos. Sim. Spearman Corr.", fontsize=20)
#plt.xticks(rotation=45)
#ax_cross.xticks(rotation=45)
#print("Got ticks", ax_multi.get_xticklabels())
#ax_multi.set_xticks(ax_multi.get_xticks(), ax_multi.get_xticklabels(), rotation=45, ha='right')
#ax_cross.set_xticks(ax_cross.get_xticks(), ax_cross.get_xticklabels(), rotation=45, ha='right')

# plt.legend(fontsize=15)

plt.savefig('multilingual_sts.png', dpi=300, bbox_inches='tight')
