import argparse
import os
os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mteb import MTEB
from sentence_transformers import SentenceTransformer


TASK_LIST = [
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
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "AskUbuntuDupQuestions",
    "SciDocs"
    "StackOverflowDupQuestions",
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

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)

    args = parser.parse_args()
    return args

def main(args):
    for task in TASK_LIST[args.startid-1:args.endid-1]:
        model_path = "/gpfswork/rech/six/commun/models/sentence-transformers_sentence-t5-base/"
        model_name = model_path.split("/")[-1].split("_")[-1]
        model = SentenceTransformer(model_path)
        evaluation = MTEB(tasks=[task])
        evaluation.run(model, output_folder=f"results/{model_name}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
