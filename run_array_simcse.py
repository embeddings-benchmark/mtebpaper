import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
import torch


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
    "CQADupstackAndroid",
    "CQADupstackEnglish",
    "CQADupstackGaming",
    "CQADupstackGisRetrieval"
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
    "SummEval",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS


class SimCSEWrapper:
    def __init__(self, modelpath="princeton-nlp/sup-simcse-bert-base-uncased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.model = AutoModel.from_pretrained(modelpath).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            # Get the embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            all_embeddings.extend(embeddings.cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--modelpath", type=str, default="/gpfswork/rech/six/commun/models/princeton-nlp/sup-simcse-bert-base-uncased")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=128)
    args = parser.parse_args()
    return args

def main(args):

    model = SimCSEWrapper(args.modelpath)

    if args.taskname is not None:
        task = args.taskname
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        model_name = args.modelpath.split("/")[-1].split("_")[-1]
        evaluation = MTEB(tasks=[task], task_langs=[args.lang], eval_splits=eval_splits)
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize)
        exit()

    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        model_name = args.modelpath.split("/")[-1].split("_")[-1]
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits)

if __name__ == "__main__":
    args = parse_args()
    main(args)
