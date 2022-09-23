import os
import random

from mteb import MTEB
from mteb.evaluation.evaluators.utils import cos_sim
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

### GLOBAL VARIABLES ###

DATAPATH = "/gpfsscratch/rech/six/commun/commun/experiments/muennighoff/mteb"

SEED = 42

K_SAMPLES = 100
LEN_KEYS = {
    "text", 
    "sentences", 
    "sentence1", 
    "sentence2",
    "sent1",
    "sent2"
    "query",
    "positive",
    "negative"
    "queries", 
    "corpus",
    "machine_summaries",
    "human_summaries",
}

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
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
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

### LOGIC ###

def get_samples_beir(hf_hub_name):
    # Somehow needs to be set in the function scope
    random.seed(SEED)
    from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader
    path = os.path.join(DATAPATH, hf_hub_name)
    print("GOT PATH", path)
    split = "validation" if "MSMARCO" in hf_hub_name else "test"
    if not os.path.exists(path):
        from beir import util
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{hf_hub_name}.zip"
        util.download_and_unzip(url, DATAPATH)
    corpus, queries, relevant_docs = BeirDataLoader(path).load(split=split)
    # Pick shortest k samples
    samples = [v["text"] + " " + v["title"] for v in sorted(list(corpus.values()), key=lambda x: len(x["text"]))[:K_SAMPLES]]
    # Optionally randomly pick
    #samples = [v["text"] + " " + v["title"] for v in random.choices(sorted(list(corpus.values()), key=lambda x: len(x["text"])), k=K_SAMPLES)]
    return samples

def load_data(hf_hub_name, subset=None):
    """
    Load dataset from Hub via cloning for easy offline usage with HF_DATASETS_OFFLINE=1
    Can be replaced with just `load_dataset(hf_hub_name, subset)` if preferred
    """
    from datasets import load_dataset
    path = os.path.join(DATAPATH, hf_hub_name)
    if os.path.exists(path):
        dataset = load_dataset(path, subset)
    else:
        from git import Repo
        Repo.clone_from("https://huggingface.co/datasets/" + hf_hub_name, path)
        dataset = load_dataset(path, subset)
    return dataset

def get_samples_ds(hf_hub_name):
    ds = load_data(hf_hub_name)
    # Optionally shuffle
    # .shuffle(seed=SEED)
    assert "test" in ds, f"No test set for {hf_hub_name}"
    len_keys = list(set(ds["test"].features.keys()) & LEN_KEYS)
    split = "test"
    k = len_keys[0]
    if isinstance(ds[split][k][0], str):
        # Select K shortest examples
        samples = sorted([x for x in ds[split][k]], key=len)[:K_SAMPLES]
    elif isinstance(ds[split][k][0], list):
        assert isinstance(ds[split][k][0][0], str), f"Too nested: {k}"
        # Select K shortest examples
        samples = [y for x in ds[split][k] for y in x]
        samples = sorted(samples, key=len)[:K_SAMPLES]
        # Optionally randomly select
        # random.choices(samples, k=K_SAMPLES)
    else:
        raise ValueError(f"Unknown type {type(ds[split][k])}")
    return samples


embeddings = {}
model = SentenceTransformer("/gpfswork/rech/six/commun/models/sentence-transformers_sentence-t5-xxl")

# Optionally custom selection
# TASKS = ["ArguAna", "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact", "Touche2020", "TRECCOVID"]

TASKS = TASK_LIST_EN


for task in MTEB(tasks=TASKS).tasks:
    print("Task: ", task)
    if "hf_hub_name" in task.description:
        hub_name = hub_url = task.description.get("hf_hub_name")
        samples = get_samples_ds(hub_name.split("/")[-1])
    if "beir_name" in task.description:
        hub_name = hub_url = "BeIR/" + task.description.get("beir_name")
        samples = get_samples_beir("/".join(hub_name.split("/")[1:]))
    embeddings[task.description["name"]] = model.encode(samples)

# Plot 1: Compute all cos sims & then average
"""
data_dict = []
for i, task_1 in enumerate(TASKS):
    data_dict.append({task_2: torch.mean(cos_sim(embeddings[task_1], embeddings[task_2])).item() for j, task_2 in enumerate(TASKS)})

data_df = pd.DataFrame(data_dict)
data_df.set_index(data_df.columns, inplace=True)


# Save
data_df.to_csv("data.csv")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(32, 16))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(data_df, dtype=np.bool))
#heatmap = sns.heatmap(data_df, mask=mask, vmin=-1, vmax=1, annot=True, cmap='Blues')
heatmap = sns.heatmap(data_df, mask=mask, vmin=data_df.values.min(), vmax=data_df.values.max(), annot=True, cmap='Blues')
heatmap.set_title('Similarity of MTEB datasets', fontdict={'fontsize':18}, pad=16)

plt.savefig('heatmap_data.png', dpi=300, bbox_inches='tight')
"""


# Plot 2: Average embeddings & then compute cos_sim

data_dict_emb = []
for i, task_1 in enumerate(TASKS):
    data_dict_emb.append({task_2: cos_sim(np.mean(embeddings[task_1], axis=0), np.mean(embeddings[task_2], axis=0)).item() for j, task_2 in enumerate(TASKS)})

data_emb_df = pd.DataFrame(data_dict_emb)
data_emb_df.set_index(data_emb_df.columns, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(36, 24))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(data_emb_df, dtype=np.bool))
heatmap = sns.heatmap(data_emb_df, mask=mask, vmin=data_emb_df.values.min(), vmax=data_emb_df.values.max(), annot=True, cmap='Blues')
#heatmap.set_title('Similarity of MTEB datasets', fontdict={'fontsize':18}, pad=16)

# Save
data_emb_df.to_csv("data.csv")
plt.savefig('heatmap_data.png', dpi=450, bbox_inches='tight')


# Plot 3: Min (/Max) embeddings & then compute cos_sim
"""
data_dict_emb = []
for i, task_1 in enumerate(TASKS):
    data_dict_emb.append({task_2: cos_sim(np.min(embeddings[i], axis=0), np.min(embeddings[j], axis=0)).item() for j, task_2 in enumerate(TASKS)})

data_emb_df = pd.DataFrame(data_dict_emb)
data_emb_df.set_index(data_emb_df.columns, inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(32, 16))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(data_emb_df, dtype=np.bool))
heatmap = sns.heatmap(data_emb_df, mask=mask, vmin=data_emb_df.values.min(), vmax=data_emb_df.values.max(), annot=True, cmap='Blues')
heatmap.set_title('Similarity of MTEB datasets', fontdict={'fontsize':18}, pad=16)

plt.savefig('heatmap_data.png', dpi=300, bbox_inches='tight')
"""
