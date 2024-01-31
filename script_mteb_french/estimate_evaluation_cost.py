import json
from typing import List
import mteb
from mteb import MTEB
import tiktoken

from utils.tasks_list import get_tasks

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

PRICE_PER_1K_TOKEN = 0.0001

TASKS_LANGS = ["fr"]


def estimate_cost_generic_task(
    task: tuple[(
        mteb.tasks.Classification,
        mteb.tasks.Clustering,
        mteb.tasks.BitextMining,
        mteb.tasks.STS,
        mteb.tasks.PairClassification
    )],
    feature_names: List[str],
    price_per_1k_tokens=0.0001,
) -> float:
    """Generic method to estimate the cost of embedding the entire dataset used to run
    a task. You need to provides the name of the features that need to be embed.

    For example: for classification tasks, the text features that are embed
    are ['sentence1', 'sentence2']

    Args:
        task (mteb.tasks.STS): the task object from mteb
        feature_names (List[str]): the names of the features that needs to be embedded
        price_per_1k_tokens (float, optional): the price for embedding 1k tokens,
            according to the API provider. Defaults to 0.0001.

    Returns:
        float: the estimated cost
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    total_n_tokens = 0
    if task.is_multilingual or task.is_crosslingual:
        for lang in task.dataset.keys():
            for split in task.description["eval_splits"]:
                for feature in feature_names:
                    total_n_tokens += sum(
                        [
                            len(tokenizer.encode(s[feature]))
                            for s in task.dataset[lang][split]
                        ]
                    )
    else:
        for split in task.description["eval_splits"]:
            for feature in feature_names:
                total_n_tokens += sum(
                    [
                        len(tokenizer.encode(s[feature]))
                        for s in task.dataset[split]
                    ]
                )

    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)

    return cost


def estimate_cost_clustering_task(
    task: mteb.tasks.Clustering,
    price_per_1k_tokens=0.0001,
) -> float:
    """Estimate the cost of embedding the entire dataset used to run
    a task of type "clustering"

    Args:
        task (mteb.tasks.Clustering): the task object from mteb
        price_per_1k_tokens (float, optional): the price for embedding 1k tokens,
            according to the API provider. Defaults to 0.0001.

    Returns:
        float: the estimated cost
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    total_n_tokens = 0
    if task.is_multilingual or task.is_crosslingual:
        for lang in task.dataset.keys():
            for split in task.description["eval_splits"]:
                for cluster in task.dataset[lang][split]["sentences"]:
                    total_n_tokens += sum(
                        [
                            len(tokenizer.encode(s))
                            for s in cluster
                        ]
                    )
    else:
        for split in task.description["eval_splits"]:
            for cluster in task.dataset[split]["sentences"]:
                total_n_tokens += sum(
                    [
                        len(tokenizer.encode(s))
                        for s in cluster
                    ]
                )

    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)

    return cost


def estimate_cost_pair_classification_task(
    task: mteb.tasks.PairClassification,
    price_per_1k_tokens=0.0001,
) -> float:
    """Estimate the cost of embedding the entire dataset used to run
    a task of type "pair_classification"

    Args:
        task (mteb.tasks.PairClassification): the task object from mteb
        price_per_1k_tokens (float, optional): the price for embedding 1k tokens,
            according to the API provider. Defaults to 0.0001.

    Returns:
        float: the estimated cost
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    total_n_tokens = 0
    if task.is_multilingual or task.is_crosslingual:
        for lang in task.dataset.keys():
            for split in task.description["eval_splits"]:
                total_n_tokens += sum(
                    [
                        len(tokenizer.encode(s))
                        for sublist in task.dataset[lang][split]["sent1"]
                        for s in sublist
                    ]
                )
                total_n_tokens += sum(
                    [
                        len(tokenizer.encode(s))
                        for sublist in task.dataset[lang][split]["sent2"]
                        for s in sublist
                    ]
                )
    else:
        for split in task.description["eval_splits"]:
            total_n_tokens += sum(
                [
                    len(tokenizer.encode(s))
                    for sublist in task.dataset[split]["sent1"]
                    for s in sublist
                ]
            )
            total_n_tokens += sum(
                [
                    len(tokenizer.encode(s))
                    for sublist in task.dataset[split]["sent2"]
                    for s in sublist
                ]
            )

    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)

    return cost


def estimate_cost_retrieval_task(
    task: mteb.tasks.Retrieval, price_per_1k_tokens=0.0001
) -> float:
    """Estimate the cost of embedding the entire dataset used to run
    a task of type "retrieval"

    Args:
        task (mteb.tasks.Retrieval): the task object from mteb
        price_per_1k_tokens (float, optional): the price for embedding 1k tokens,
            according to the API provider. Defaults to 0.0001.

    Returns:
        float: the estimated cost
    """
    # we might not know which is the encoder for each model. So we use the base one from ADA
    # it leads to approx 2.5 tokens/word (which is high), the the cost may be overestimated
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    total_n_tokens = 0
    for split in task.queries.keys():
        if split in task.description["eval_splits"]:
            queries_encodings_lengths = sum(
                [len(tokenizer.encode(s)) for s in task.queries[split].values()]
            )
            documents_encodings_lengths = sum(
                [len(tokenizer.encode(s["text"])) for s in task.corpus[split].values()]
            )
            total_n_tokens += queries_encodings_lengths + documents_encodings_lengths
    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)

    return cost


def estimate_cost_reranking_task(
    task: mteb.tasks.Reranking, price_per_1k_tokens=0.0001
) -> float:
    """Estimate the cost of embedding the entire dataset used to run
    a task of type "reranking"

    Args:
        task (mteb.tasks.Reranking): the task object from mteb
        price_per_1k_tokens (float, optional): the price for embedding 1k tokens,
            according to the API provider. Defaults to 0.0001.

    Returns:
        float: the estimated cost
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    encodings_lengths = task.dataset.map(
        lambda x: {
            "n_tokens": len(tokenizer.encode(x["query"]))
            + sum([len(tokenizer.encode(s)) for s in x["positive"]])
            + sum([len(tokenizer.encode(s)) for s in x["negative"]])
        }
    )

    total_n_tokens = 0
    for split in task.description["eval_splits"]:
        total_n_tokens += sum(encodings_lengths[split]["n_tokens"])
    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)

    return cost


def estimate_cost_Summarization_task(
    task: mteb.tasks.Summarization, price_per_1k_tokens=0.0001
) -> float:
    """Estimate the cost of embedding the entire dataset used to run
    a task of type "Summarization"

    Args:
        task (mteb.tasks.Summerization): the task object from mteb
        price_per_1k_tokens (float, optional): the price for embedding 1k tokens,
            according to the API provider. Defaults to 0.0001.

    Returns:
        float: the estimated cost
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    task.load_data()
    total_n_tokens = 0
    tokenized = task.dataset.map(
        lambda x: {
            "n_tokens": len(tokenizer.encode(x["text"]))
            + sum([len(tokenizer.encode(s)) for s in x["human_summaries"]])
            + sum([len(tokenizer.encode(s)) for s in x["machine_summaries"]])
        }
    )
    if task.is_multilingual or task.is_crosslingual:
        for lang in task.dataset.keys():
            for split in task.description["eval_splits"]:
                total_n_tokens += sum(tokenized[lang][split]["n_tokens"])
    else:
        for split in task.description["eval_splits"]:
            total_n_tokens += sum(tokenized[split]["n_tokens"])

    cost = round(total_n_tokens / 1000 * price_per_1k_tokens, 2)

    return cost


TASK_LIST = [task_name for _, task_name in get_tasks()]

costs_dump = {"PRICE_FOR_1K_TOKENS_IN_$": PRICE_PER_1K_TOKEN, "PRICE_FOR_TASK_EVALUATION": {}}
evaluation = MTEB(tasks=TASK_LIST, task_langs=TASKS_LANGS)
for task in evaluation.tasks:
    match task.description["type"]:
        case "BitextMining":
            cost = estimate_cost_generic_task(
                task, ["sentence1", "sentence2"], PRICE_PER_1K_TOKEN
            )
        case "Classification":
            cost = estimate_cost_generic_task(task, ["text"], PRICE_PER_1K_TOKEN)
        case "Clustering":
            cost = estimate_cost_clustering_task(task, PRICE_PER_1K_TOKEN)
        case "PairClassification":
            cost = estimate_cost_pair_classification_task(
                task, PRICE_PER_1K_TOKEN
            )
        case "Retrieval":
            cost = estimate_cost_retrieval_task(task, PRICE_PER_1K_TOKEN)
        case "Reranking":
            cost = estimate_cost_reranking_task(task, PRICE_PER_1K_TOKEN)
        case "STS":
            cost = estimate_cost_generic_task(
                task, ["sentence1", "sentence2"], PRICE_PER_1K_TOKEN
            )
        case "Summarization":
            cost = estimate_cost_Summarization_task(task, PRICE_PER_1K_TOKEN)

    print(task.description["name"], ":", cost, "$")
    costs_dump["PRICE_FOR_TASK_EVALUATION"][task.description["name"]] = cost

with open("cost_estimation.json", "w") as f:
    json.dump(costs_dump, f)
