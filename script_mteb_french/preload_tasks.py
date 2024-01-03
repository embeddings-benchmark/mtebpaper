import os
from mteb import MTEB

"""Downloads all MTEB tasks"""

TASK_LIST_CLASSIFICATION = [
    "AmazonReviewsClassification",
    "MasakhaNEWSClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
]

TASK_LIST_CLUSTERING = [
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "HALClusteringS2S",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringS2S",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "OpusparcusPC",
]

TASK_LIST_RERANKING = ["SyntecReranking", "AlloprofReranking"]

TASK_LIST_RETRIEVAL = [
    "AlloprofRetrieval",
    "BSARDRetrieval",
    "HagridRetrieval",
    "SyntecRetrieval",
]

TASK_LIST_STS = ["STSBenchmarkMultilingualSTS", "STS22"]

TASK_LIST_SUMMARIZATION = [
    "SummEvalFr",
]

TASK_LIST_BITEXTMINING = [
    "DiaBLaBitextMining",
    "FloresBitextMining",
]


TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
    + TASK_LIST_BITEXTMINING
)

os.environ["HF_DATASETS_OFFLINE"] = "0"  # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 1 for offline
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/mathi/.cache/huggingface/models"
os.environ["HF_DATASETS_CACHE"] = "C:/Users/mathi/.cache/huggingface/datasets"
os.environ["HF_MODULES_CACHE"] = "C:/Users/mathi/.cache/huggingface/modules"
os.environ["HF_METRICS_CACHE"] = "C:/Users/mathi/.cache/huggingface/metrics"

evaluation = MTEB(tasks=TASK_LIST, task_langs=["fr"])

for task in evaluation.tasks:
    task.load_data()
