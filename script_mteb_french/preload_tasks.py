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

TASK_LIST_RERANKING = [
    "SyntecReranking",
    "AlloprofReranking"
]

TASK_LIST_RETRIEVAL = [
    "AlloprofRetrieval", 
    "BSARDRetrieval", 
    "SyntecRetrieval"
]

TASK_LIST_STS = [
    "STSBenchmarkMultilingualSTS",
    "STS22",
    "SICKFr"
]

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

evaluation = MTEB(tasks=TASK_LIST, task_langs=["fr"])

for task in evaluation.tasks:
    task.load_data()
