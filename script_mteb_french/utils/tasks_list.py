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

TASK_LIST_RETRIEVAL = ["AlloprofRetrieval", "BSARDRetrieval", "SyntecRetrieval"]

TASK_LIST_STS = ["STSBenchmarkMultilingualSTS", "STS22", "SICKFr"]

TASK_LIST_SUMMARIZATION = [
    "SummEvalFr",
]

TASK_LIST_BITEXTMINING = [ "DiaBLaBitextMining", "FloresBitextMining"]

TASKS = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
    + TASK_LIST_BITEXTMINING
)

TYPES_TO_TASKS = {
    "all": TASKS,
    "classification": TASK_LIST_CLASSIFICATION,
    "clustering": TASK_LIST_CLUSTERING,
    "reranking": TASK_LIST_RERANKING,
    "retrieval": TASK_LIST_RETRIEVAL,
    "pair_classification": TASK_LIST_PAIR_CLASSIFICATION,
    "sts": TASK_LIST_STS,
    "summarization": TASK_LIST_SUMMARIZATION,
    "bitextmining": TASK_LIST_BITEXTMINING,
}


def get_tasks(task_types:list[str]=["all"]) -> list[tuple[str, str]]:
    """Get the list of task based on task types

    Args:
        task_types (list[str], optional): the type of tasks. Defaults to ["all"].

    Returns:
        list[tuple[str, str]]: the list of tuple (task_type, task_name)
    """
    if not isinstance(task_types, list):
        task_types = [task_types]
    assert all([t in TYPES_TO_TASKS.keys() for t in task_types]), (
        f"All types provided in 'task_type' argument must be in {list(TYPES_TO_TASKS.keys())}. Got {task_types}"
    )
    return [
        (task_type, task)
        for task_type in task_types
        for task in TYPES_TO_TASKS[task_type]
    ]