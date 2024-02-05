
import logging
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import mteb
from mteb import MTEB
from mteb.evaluation.evaluators.utils import cos_sim
from mteb import LangMapping 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import hsv_to_rgb
import seaborn as sns

from ..utils.tasks_list import get_tasks, TYPES_TO_TASKS


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
        samples = np.concatenate(samples[:100]).flatten()

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


def get_all_samples(tasks:list[str], n_samples:int=90, langs:list[str]="fr") -> dict[str, list[str]]:
    """Get the samples from the dataset of each task

    Args:
        tasks (list[str]): list of task names
        n_samples (int, optional): the mount of samples to get. Defaults to 90.
        langs (list[str], optional): the langs from which the samples are taken. Defaults to "fr".

    Returns:
        tuple[list[str], list[str]]: a dict with {"task name" : [tasks samples], ...}
    """

    task_types = ["bitextmining", "classification", "clustering", "pairclassification", "reranking", "retrieval", "sts", "summarization"]
    text_keys_of_tasks = ["sentence1", "text", "sentences", "sent1", "negative", "text", "sentence1", "human_summaries"]
    task_type_to_text_key = dict(zip(task_types, text_keys_of_tasks))

    task_samples_dict = {}
    evaluation = MTEB(tasks=tasks, task_langs=langs)
    for task in tqdm(evaluation.tasks):
        task.load_data(eval_splits=task.description.get("eval_splits", []))
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

        task_samples_dict[task.description["name"]] = samples

    assert all([len(ts) == n_samples for ts in list(task_samples_dict.values())]), (
        "A task doesn't have the required number of samples :"
        f"{list(task_samples_dict.keys())[[len(ts) == n_samples for ts in list(task_samples_dict.values())].index(False)]}",
    )

    return task_samples_dict


def get_embeddings(samples:list[str]) -> np.array:
    """Gets the embeddings of a list of strings

    Args:
        samples (list[str]): a list of samples (=strings)
        model_name (str, optional): name of the sentence-transformer.
            Defaults to "intfloat/multilingual-e5-large".

    Returns:
        np.array: the embeddings
    """
    embeddings = MODEL.encode(samples)

    return embeddings


def parse_args() -> Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default="all")
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

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    _EXTENDED_LANGS = _extend_lang_code(args.langs) + ["fr-en"]
    RNG = np.random.default_rng(seed=args.seed)
    TASK_LIST = get_tasks(args.task_type)
    TASK_NAMES = [name for _, name in TASK_LIST]
    MODEL = SentenceTransformer(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    # get samples and embeddings
    logging.info(f"Getting samples for {len(TASK_NAMES)} tasks...")
    tasks_samples_dict = get_all_samples(TASK_NAMES, args.n_samples, _EXTENDED_LANGS)
    logging.info(f"Embedding samples...")
    tasks_embeddings_dict = {k: get_embeddings(v) for k, v in tasks_samples_dict.items()}

    # Run PCA
    logging.info(f"Computing PCA...")
    pca = PCA(n_components=10)
    embeddings = [i for j in list(tasks_embeddings_dict.values()) for i in j]
    reduced = pca.fit_transform(embeddings)

    # Plot explained variance
    plt.plot(pca.explained_variance_ratio_)
    plt.title("PCA - Explained variance ratio")
    plt.savefig(os.path.join(args.output_folder, f"PCA_explained_variance_ratio_{args.task_type}.png"), bbox_inches='tight')

    # Plot PCA components
    types_to_colors = {
        k : [hsv_to_rgb(((j/1.5)/len(TYPES_TO_TASKS), i, 1))
             for i in np.arange(.3, 1, .8/len(v))]
        for j, (k,v) in enumerate(TYPES_TO_TASKS.items())
    }
    task_name_to_color = {
        i:j
        for k, v in TYPES_TO_TASKS.items()
        for i, j in zip(v, types_to_colors[k])
    }
    labels = np.concatenate([np.full((args.n_samples), i) for i in range(len(tasks_samples_dict))]).flatten()
    plt.figure(figsize=(36, 24))
    fig, ax = plt.subplots()
    for i, name in enumerate(tasks_embeddings_dict.keys()):
        color = task_name_to_color[name]
        x = reduced[i*args.n_samples:(i+1)*args.n_samples, 0]
        y = reduced[i*args.n_samples:(i+1)*args.n_samples, 1]
        centroid = (x.mean(), y.mean())
        ax.scatter(x, y, s=.5, color=color)
        ax.scatter(centroid[0], centroid[1], lw=1, s=50, color=color, label=f"{i+1} - {name}", edgecolors="black")
        ellipse = Ellipse(xy=centroid, width=x.std(), height=y.std(), fc=color, lw=0, alpha=.3)
        ax.add_patch(ellipse)
    # Setup legend on the side
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(False)
    plt.savefig(os.path.join(args.output_folder, f"PCA_components_{args.task_type}.png"), bbox_inches='tight')
    logging.info(f"PCA plots done !")

    # Plot Averaged embeddings cosine similarities
    data_dict_emb = []
    for i, task_1 in enumerate(tasks_embeddings_dict.keys()):
        data_dict_emb.append({task_2: cos_sim(np.mean(tasks_embeddings_dict[task_1], axis=0), np.mean(tasks_embeddings_dict[task_2], axis=0)).item() for task_2 in tasks_embeddings_dict.keys()})

    data_emb_df = pd.DataFrame(data_dict_emb)
    data_emb_df.set_index(data_emb_df.columns, inplace=True)
    plt.figure(figsize=(36, 24))
    # define the mask to set the values in the upper triangle to True
    mask = np.triu(np.ones_like(data_emb_df, dtype=bool))
    heatmap = sns.heatmap(data_emb_df, mask=mask, vmin=data_emb_df.values.min(), vmax=data_emb_df.values.max(), annot=True, cmap='Blues')
    plt.savefig(os.path.join(args.output_folder, f'cosim_{args.task_type}.png'), dpi=300, bbox_inches='tight')
