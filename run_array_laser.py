"""
See https://github.com/facebookresearch/LASER/issues/211
"""

import argparse
import logging
import os

import numpy as np
import subprocess

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mteb import MTEB
from sentence_transformers import SentenceTransformer

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
#    "RedditClusteringP2P",
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
    "SciDocs",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
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
    "SummEval",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS


#with open("LASER_script.sh", "w") as f:
#    f.write("LASER=/content/LASER ./LASER/tasks/embed/embed.sh tmp.txt tmp.bin")
# Run `chmod u+rx LASER_script.sh` to give permissions
# !chmod u+rx LASER_script.sh

class LASER():
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if os.path.exists("tmp.txt"):
            os.remove("tmp.txt")
        if os.path.exists("tmp.bin"):
            os.remove("tmp.bin")

        # LASER expects one text per line, so we need to replace newlines
        sentences = [s.replace("\n", " ") for s in sentences]
        with open("tmp.txt", "w") as f:
            f.write("\n".join(sentences))
        
        print(len(sentences))
        rc = subprocess.call("./LASER_script.sh", shell=True)
              
        dim = 1024
        X = np.fromfile("tmp.bin", dtype=np.float32, count=-1)                                                                          
        X.resize(X.shape[0] // dim, dim)
        print(X.shape)
        return X

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=128)
    args = parser.parse_args()
    return args

def main(args):

    model = LASER()
    model_name = "LASER2"

    if args.taskname is not None:
        task = args.taskname
        evaluation = MTEB(tasks=[task], task_langs=[args.lang], eval_splits=["test"])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize)
        exit()

    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        evaluation = MTEB(tasks=[task], task_langs=[args.lang], eval_splits=["test"])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize)

if __name__ == "__main__":
    args = parse_args()
    main(args)

