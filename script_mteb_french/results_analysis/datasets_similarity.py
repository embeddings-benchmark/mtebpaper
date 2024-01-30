
import logging
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import mteb
from mteb import MTEB
from mteb import LangMapping 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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

TASK_LIST_BITEXTMINING = [
    "DiaBLaBitextMining",
    "FloresBitextMining",
]

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


def get_samples_from_dataset(
    task: tuple[(
        mteb.tasks.Classification,
        mteb.tasks.Clustering,
        mteb.tasks.BitextMining,
        mteb.tasks.STS,
        mteb.tasks.PairClassification
    )],
    text_key:str,
    n_samples:int=90,
    langs:list[str]=["fr"],
    ) -> list[str]:
    """Gets a specific amount of random samples (i.e. texts)
    from a task's dataset

    Args:
        task (mteb.tasks): the MTEB task
        text_key (str): the key of the dataset that contains the text
        n_samples (int, optional): the mount of samples to get. Defaults to 90.
        langs (str, optional): the languages to use for multilingual datasets. Defaults to ["fr"].

    Returns:
        list[str]: a list of texts from the task's dataset
    """
    samples = []
    dataset = task.dataset if task.description["type"].lower() != "retrieval" else task.corpus
    if task.is_multilingual or task.is_crosslingual:
        for lang in langs:
            if lang in dataset:
                for split in task.description["eval_splits"]:
                    samples.extend(dataset[lang][split][text_key])
    else:
        for split in task.description["eval_splits"]:
            samples.extend(dataset[split][text_key])
    # flatten in case we have a list of list of strings (e.g. clustering tasks)
    if isinstance(samples[0], list):
        samples = np.concatenate(samples).flatten()

    return RNG.choice(samples, size=n_samples, replace=False)


def get_samples_from_retrieval_dataset(
    task: tuple[(
        mteb.tasks.Classification,
        mteb.tasks.Clustering,
        mteb.tasks.BitextMining,
        mteb.tasks.STS,
        mteb.tasks.PairClassification
    )],
    n_samples:int=90,
    ) -> list[str]:
    """Gets a specific amount of random samples (i.e. texts)
    from a task's dataset

    Args:
        task (mteb.tasks): the MTEB task
        n_samples (int, optional): the mount of samples to get. Defaults to 90.

    Returns:
        list[str]: a list of texts from the task's dataset
    """
    samples = []
    for split in task.description["eval_splits"]:
        data = list(task.corpus[split].values())
        data = [t["text"] for t in data]
        samples.extend(data)

    return RNG.choice(samples, size=n_samples, replace=False)


def _extend_lang_code(langs):
    total_langs = langs.copy()
    for lang in langs:
        if LangMapping.LANG_MAPPING.get(lang) is not None:
            total_langs.extend(LangMapping.LANG_MAPPING[lang])

    return total_langs


def get_all_samples(tasks:list[str], n_samples:int=90, langs:list[str]="fr") -> tuple[list[str], list[str]]:

    task_types = ["bitextmining", "classification", "clustering", "pair_classification", "reranking", "retrieval", "sts", "summarization"]
    text_keys_of_tasks = ["sentence1", "text", "sentences", "sent1", "negatives", "text", "sentence1", "human_summaries"]
    task_type_to_text_key = dict(zip(task_types, text_keys_of_tasks))

    tasks_names = []
    tasks_samples = []
    evaluation = MTEB(tasks=tasks, task_langs=langs)
    for task in tqdm(evaluation.tasks):
        task.load_data()
        text_key = task_type_to_text_key[task.description["type"].lower()]
        match task.description["type"].lower():
            case "retrieval":
                samples = get_samples_from_retrieval_dataset(task, n_samples)
            case other:
                samples = get_samples_from_dataset(
                    task,
                    text_key=text_key,
                    n_samples=n_samples,
                    langs=langs
                    )

        tasks_names.append(task.description["name"])
        tasks_samples.append(samples)

    assert all([len(ts) == n_samples for ts in tasks_samples]), (
        "A task doesn't have the required number of samples :"
        f"{tasks_names[[len(ts) == n_samples for ts in tasks_samples].index(False)]}",
    )

    return tasks_names, tasks_samples


def get_embeddings(samples:list[str], model_name:str="intfloat/multilingual-e5-large") -> np.array:
    """Gets the embeddings of a list of strings

    Args:
        samples (list[str]): a list of samples (=strings)
        model_name (str, optional): name of the sentence-transformer.
            Defaults to "intfloat/multilingual-e5-large".

    Returns:
        np.array: the embeddings
    """
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = np.concatenate([model.encode(s) for s in tqdm(samples)])

    return embeddings


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, choices=list(TYPES_TO_TASKS.keys()), default="all")
    parser.add_argument("--langs", type=list[str], default=["fr"])
    parser.add_argument("--output_folder", type=str, default="./datasets_similarity_analysis")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=90)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()
    print(torch.cuda.is_available())

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    _EXTENDED_LANGS = _extend_lang_code(args.langs) + ["fr-en"]
    RNG = np.random.default_rng(seed=args.seed)

    # get samples and embeddings
    logging.info(f"Getting samples for {len(TYPES_TO_TASKS[args.task_type])} tasks...")
    tasks_names, tasks_samples = get_all_samples(TYPES_TO_TASKS[args.task_type], args.n_samples, _EXTENDED_LANGS)
    logging.info(f"Embedding samples...")
    embeddings = get_embeddings(tasks_samples, args.model_name)

    # Run PCA
    logging.info(f"Computing PCA...")
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(embeddings)

    # Plot explained variance
    plt.plot(pca.explained_variance_ratio_)
    plt.title("PCA - Explained variance ratio")
    plt.savefig(os.path.join(args.output_folder, f"PCA_explained_variance_ratio_{args.task_type}.png"), bbox_inches='tight')

    # Plot PCA components
    labels = np.concatenate([np.full((args.n_samples), i) for i in range(len(tasks_samples))]).flatten()
    fig, ax = plt.subplots()
    cmap = matplotlib.colormaps["plasma"]
    n_colors = len(tasks_names)
    colors = [cmap(i) for i in np.arange(0, 1, 1/n_colors)]
    for i, (name, color) in enumerate(zip(tasks_names, colors)):
        x = reduced[i*args.n_samples:(i+1)*args.n_samples, 0]
        y = reduced[i*args.n_samples:(i+1)*args.n_samples, 1]
        centroid = (x.mean(), y.mean())
        ax.scatter(x, y, s=1, color=color)
        ax.scatter(centroid[0], centroid[1], lw=2, s=50, color=color, label=name)
        ellipse = Ellipse(xy=centroid, width=x.std(), height=y.std(), fc=color, lw=0, alpha=.2)
        ax.add_patch(ellipse)
    # Setup legend on the side
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(False)

    plt.savefig(os.path.join(args.output_folder, f"PCA_components_{args.task_type}.png"), bbox_inches='tight')
    logging.info(f"PCA plots done !")
