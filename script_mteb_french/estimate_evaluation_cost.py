import mteb
from mteb import MTEB
import tiktoken

"""
This script is used to evaluate the cost of evaluating a task.

It counts the total number of tokens of the dataset, and computes the cost based on the api price

You need to setup 3 parameters :
- TASK_LIST: The list of tasks you want to evaluate
- PRICE_PER_1K_TOKEN : The price per 1000 tokens of the api you are about to use
- TASKS_LANGS: The language on which you want to evaluate

NOTICE:
- As we might not know which tokenizer is used by the model api,
we are using clk100_base from tiktoken by default. (approx 2.5 tokens/words)
- We do not take into account that the benchmark uses static storage with chromaDB
=> Overall the output costs may likely be overestimated
"""


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

PRICE_PER_1K_TOKEN = 0.0001

TASKS_LANGS = ["fr"]

# TODO : add functions for cost estimation for other task types
# TODO : add argparser
# TODO: dump result in a file ?


def estimate_cost_retrieval_task(
    task: mteb.tasks.Retrieval, price_per_1k_tokens=0.0001
):
    """Evaluates the cost for a retrieval task

    Args:
        task (mteb.tasks.Retrieval): the task object of mteb
        price_per_1k_tokens (float, optional): Price for 1000 tokens. Defaults to 0.0001.

    Returns:
        _type_: _description_
    """
    # we might not know which is the encoder for each model. So we use the base one from ADA
    # it leads to approx 2.5 tokens/word (which is high), the the cost may be overestimated
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    total_n_tokens = 0
    for key in task.queries.keys():
        queries_encodings_lengths = sum(
            [len(tokenizer.encode(s)) for s in task.queries[key].values()]
        )
        documents_encodings_lengths = sum(
            [len(tokenizer.encode(s["text"])) for s in task.corpus[key].values()]
        )
        total_n_tokens += queries_encodings_lengths + documents_encodings_lengths
    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)
    print(task.description["name"], ":", cost, "$")

    return cost


def estimate_cost_reranking_task(
    task: mteb.tasks.Retrieval, price_per_1k_tokens=0.0001
):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    encodings_lengths = task.dataset.map(
        lambda x: {
            "n_tokens": len(tokenizer.encode(x["query"]))
            + sum([len(tokenizer.encode(s)) for s in x["positive"]])
            + sum([len(tokenizer.encode(s)) for s in x["negative"]])
        }
    )
    cost = round(
        sum(encodings_lengths["test"]["n_tokens"]) / 1000 * price_per_1k_tokens, 2
    )
    print(task.description["name"], ":", cost, "$")

    return cost


evaluation = MTEB(tasks=TASK_LIST, task_langs=TASKS_LANGS)
for task in evaluation.tasks:
    print(task.description["name"])
    match task.description["type"]:
        case "Retrieval":
            _ = estimate_cost_retrieval_task(task, PRICE_PER_1K_TOKEN)
        case "Reranking":
            _ = estimate_cost_reranking_task(task, PRICE_PER_1K_TOKEN)
