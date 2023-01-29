import argparse
import logging
import os
from typing import Dict, List, Union

from mteb import MTEB
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.multiprocessing as mp
from torch import Tensor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
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


class SentenceTransformerSpecb(SentenceTransformer):
    # Requires: 
    # https://github.com/Muennighoff/sentence-transformers/tree/sgpt_poolings_specb
    # pip install git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokens = ["[SOS]", "{SOS}"]
        self.sep = " "
        self._first_module().tokenizer.add_tokens(tokens, special_tokens=True)
        self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        self._first_module().bos_spec_token_q = self._first_module().tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d = self._first_module().tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_q_rep = self._first_module().tokenizer.encode("[", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_q = self._first_module().tokenizer.encode("]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d_rep = self._first_module().tokenizer.encode("{", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_d = self._first_module().tokenizer.encode("}", add_special_tokens=False)[0]
        self._first_module().replace_bos = True

    def encode(self, sentences, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # Add specb query token
        sentences = ["[SOS]" + sent for sent in sentences]
        return super().encode(sentences, **kwargs)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # Will be replaced with [ in the models tokenization
        # If we would put [ here, there is a risk of it getting chained with a different token when encoding
        queries = ["[SOS]" + q for q in queries]
        return super().encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        # Will be replaced with { in the models tokenization
        # If we would put { here, there is a risk of it getting chained with a different token when encoding
        sentences = [("{SOS}" + doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else "{SOS}" + doc["text"].strip() for doc in corpus]
        return super().encode(sentences, batch_size=batch_size, **kwargs)

    def encode_corpus_parallel(
        self, corpus: List[Dict[str, str]], pool: Dict[str, object], batch_size: int, chunk_id: int, **kwargs
    ):
        if type(corpus) is dict:
            sentences = [
                ("{SOS}" + corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else "{SOS}" + corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                ("{SOS}" + doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else "{SOS}" + doc["text"].strip()
                for doc in corpus
            ]

        if chunk_id is not None and chunk_id >= len(pool["processes"]):
            output_queue = pool["output"]
            output_queue.get()

        input_queue = pool["input"]
        input_queue.put([chunk_id, batch_size, sentences])


    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(
                target=SentenceTransformer._encode_multi_process_worker,
                args=(process_id, device_name, self.model, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool["output"]
        [output_queue.get() for _ in range(len(pool["processes"]))]
        return self.model.stop_multi_process_pool(pool)


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--addspecbdoc", action='store_true')
    parser.add_argument("--addspecbquery", action='store_true')
    parser.add_argument("--modelpath", type=str, default="/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-msmarco-specb-bitfit")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=128)
    args = parser.parse_args()
    return args

def main(args):

    if args.addspecbdoc or args.addspecbquery:
        model = SentenceTransformerSpecb(args.modelpath) # Only used for SGPT-msmarco models
    else:
        model = SentenceTransformer(args.modelpath)

    if args.taskname is not None:
        task = args.taskname
        model_name = args.modelpath.split("/")[-1].split("_")[-1]
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits)
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
