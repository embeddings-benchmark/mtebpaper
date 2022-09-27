import io
import json
import os
import sys

from mteb import MTEB

### GLOBAL VARIABLES ###

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
    "CQADupstackRetrieval",
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
]


TASK_LIST_SUMMARIZATION = [
    "SummEval",
]

TASK_LIST_EN = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
)


TASK_LIST_NAMES = [
    ("Class.", TASK_LIST_CLASSIFICATION, ["en", "en-en"]),
    ("Clust.", TASK_LIST_CLUSTERING, ["en", "en-en"]),
    ("PairClass.", TASK_LIST_PAIR_CLASSIFICATION, ["en", "en-en"]),
    ("Rerank.", TASK_LIST_RERANKING, ["en", "en-en"]),
    ("Retr.", TASK_LIST_RETRIEVAL, ["en", "en-en"]),
    ("STS", TASK_LIST_STS, ["en", "en-en"]),
    ("Summ.", TASK_LIST_SUMMARIZATION, ["en", "en-en"]),
    # ("BitextMining", TASK_LIST_BITEXT, []),
    ("Avg.", TASK_LIST_EN, ["en", "en-en"]),
]

SELFSUPERVISED_MODELS = [
    "glove.6B.300d",
    "komninos",
    "LASER2",
    "LaBSE",
    "bert-base-uncased",
    "msmarco-bert-co-condensor",
    "allenai-specter",
    "unsup-simcse-bert-base-uncased",
]

SUPERVISED_MODELS = [
    "sup-simcse-bert-base-uncased",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "contriever-base-msmarco",
    "text-similarity-ada-001",
    "SGPT-125M-weightedmean-nli-bitfit",
    "SGPT-5.8B-weightedmean-nli-bitfit",
    "SGPT-125M-weightedmean-msmarco-specb-bitfit",
    "SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    "SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    "SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    "sgpt-bloom-7b1-msmarco",
    "gtr-t5-base", # 110M
    "gtr-t5-large",
    "gtr-t5-xl",
    "gtr-t5-xxl", # 4.8B
    "sentence-t5-base", # 110M
    "sentence-t5-large", # 110M
    "sentence-t5-xl", # 110M
    "sentence-t5-xxl", # 4.8B
]

MODEL_TO_NAME = {
    "bert-base-uncased": "BERT",
    "gtr-t5-base": "GTR-Base",
    "gtr-t5-large": "GTR-Large",
    "gtr-t5-xl": "GTR-XL",
    "gtr-t5-xxl": "GTR-XXL",
    "sentence-t5-base": "ST5-Base",
    "sentence-t5-large": "ST5-Large",
    "sentence-t5-xl": "ST5-XL",
    "sentence-t5-xxl": "ST5-XXL",
    "SGPT-125M-weightedmean-msmarco-specb-bitfit": "SGPT-125M-msmarco",
    "SGPT-1.3B-weightedmean-msmarco-specb-bitfit": "SGPT-1.3B-msmarco",
    "SGPT-2.7B-weightedmean-msmarco-specb-bitfit": "SGPT-2.7B-msmarco",
    "SGPT-5.8B-weightedmean-msmarco-specb-bitfit": "SGPT-5.8B-msmarco",
    "sgpt-bloom-7b1-msmarco": "SGPT-BLOOM-7.1B-msmarco",
    "SGPT-125M-weightedmean-nli-bitfit": "SGPT-125M-nli",
    "SGPT-5.8B-weightedmean-nli-bitfit": "SGPT-5.8B-nli",
    "sup-simcse-bert-base-uncased": "SimCSE-BERT-sup",
    "contriever-base-msmarco": "Contriever",
    "msmarco-bert-co-condensor": "coCondenser-msmarco", # They write it as coCondenser in the paper
    "unsup-simcse-bert-base-uncased": "SimCSE-BERT-unsup",
    "glove.6B.300d": "Glove",
    "komninos": "Komninos",
    "all-MiniLM-L6-v2": "MiniLM-L6",
    "all-MiniLM-L12-v2": "MiniLM-L12",
    "paraphrase-multilingual-MiniLM-L12-v2": "MiniLM-L12-multilingual",
    "all-mpnet-base-v2": "MPNet",
    "paraphrase-multilingual-mpnet-base-v2": "MPNet-multilingual",
    "allenai-specter": "SPECTER",
    "text-similarity-ada-001": "Ada Similarity",
}


### LOGIC ###

results_folder = sys.argv[1].strip("/")
all_results = {}

for model_name in os.listdir(results_folder):
    model_res_folder = os.path.join(results_folder, model_name)
    if os.path.isdir(model_res_folder):
        all_results.setdefault(model_name, {})
        for file_name in os.listdir(model_res_folder):
            if not file_name.endswith(".json"):
                print(f"Skipping non-json {file_name}")
                continue
            with io.open(os.path.join(model_res_folder, file_name), "r", encoding="utf-8") as f:
                results = json.load(f)
                all_results[model_name] = {**all_results[model_name], **{file_name.replace(".json", ""): results}}

def get_row(dataset, model_name, limit_langs=[], skip_langs=[]):
    # CQADupstackRetrieval uses the same metric as its subsets
    tasks = MTEB(tasks=[dataset.replace("CQADupstackRetrieval", "CQADupstackTexRetrieval")]).tasks
    assert len(tasks) == 1, f"Found {len(tasks)} for {dataset}. Expected 1."
    main_metric = tasks[0].description["main_score"]
    test_result = all_results.get(model_name, {}). get(dataset, {})

    # Dev / Val set is used for MSMARCO (See BEIR paper)
    if "MSMARCO" in dataset:
        test_result = (
            test_result.get("dev") if "dev" in test_result else test_result.get("validation")
        )
    else:
        test_result = test_result.get("test")

    for lang in tasks[0].description["eval_langs"]:
        if (limit_langs and lang not in limit_langs) or (skip_langs and lang in skip_langs):
            continue
        elif test_result is None:
            raise NotImplementedError(f"Got no test result {test_result} for ds: {dataset} model: {model_name}")

        test_result_lang = test_result.get(lang, test_result)
        if main_metric == "cosine_spearman":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("spearman")
        elif main_metric == "ap":
            test_result_lang = test_result_lang.get("cos_sim", {}).get("ap")
        else:
            test_result_lang = test_result_lang.get(main_metric)

        if test_result_lang is None:
            raise NotImplementedError

        return test_result_lang
    raise NotImplementedError


table = "Task ($\ rightarrow$) & " + " & ".join([x[0] for x in TASK_LIST_NAMES]) + " \\\\" + "\n"
table += "Num. Datasets ($\ rightarrow$) & " + " & ".join([str(len(x[1])) for x in TASK_LIST_NAMES]) + " \\\\" + "\n"
table += "Model ($\downarrow$) & " + " & ".join([x[0] for x in TASK_LIST_NAMES]) + " \\\\" + "\n"


def add_to_table(model_list, table):
    for model in model_list:
        results = []
        for (task_name, task_list, limit_langs) in TASK_LIST_NAMES:
            try:
                model_task_results = [get_row(task, model, limit_langs=limit_langs) for task in task_list]
            except:
                results.append("")
                continue
            results.append(str(round(100 * (sum(model_task_results) / len(model_task_results)), 2)))

        model_name = MODEL_TO_NAME.get(model, model)
        table += model_name + " & " + " & ".join(results) + " \\\\" + "\n"
    return table


table = add_to_table(SELFSUPERVISED_MODELS, table)
table = add_to_table(SUPERVISED_MODELS, table)

with open("avg_table.txt", "w") as f:
    f.write(table)

