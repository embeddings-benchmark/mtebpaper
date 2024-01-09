import argparse

from mteb import MTEB

from src.ModelConfig import ModelConfig

"""
How to use ?
------------

You need 2 things (=variables) :

- TASKS : a list of tasks available in the MTEB module
=> it is a list of strings corresponding to the tasks names
Example : TASKS = [SyntecRetrieval]

- MODELS : a list of model configs (=ModelConfig objects)
=> each model config must be provided a model name and a model_type.
=> supported model_type are "sentence_transformer", "voyage_ai", "open_ai"
Example: MODELS = [ModelConfig("intfloat/multilingual-e5-base", model_type="sentence_transformer")]
"""

#############################
# Step 1 : Setup model list #
#############################
SENTENCE_TRANSORMER_MODELS = [
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "flaubert/flaubert_base_uncased",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    "dangvantuan/sentence-camembert-base",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "distilbert-base-uncased",
    "Geotrend/distilbert-base-25lang-cased",
    "Geotrend/distilbert-base-en-fr-es-pt-it-cased",
    "Geotrend/distilbert-base-en-fr-cased",
    "Geotrend/distilbert-base-fr-cased",
    "Geotrend/bert-base-25lang-cased",
    "Geotrend/bert-base-15lang-cased",
    "Geotrend/bert-base-10lang-cased",
    "shibing624/text2vec-base-multilingual",
    "izhx/udever-bloom-560m",
    "izhx/udever-bloom-1b1",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "sentence-transformers/sentence-t5-xl",
    "sentence-transformers/sentence-t5-xxl",
]

"""
SENTENCE_TRANSORMER_MODELS = [
    "camembert/camembert-base", # bug
    "camembert/camembert-large", # bug
    "dangvantuan/sentence-camembert-large", # bug
    "izhx/udever-bloom-3b", # too big
    "izhx/udever-bloom-7b1", # too big
    "intfloat/e5-mistral-7b-instruct", # too big
]
"""

LASER_MODELS = ["Laser2"]

VOYAGE_MODELS = ["voyage-lite-01", "voyage-01"]

OPEN_AI_MODELS = ["text-embedding-ada-002"]

MODELS = [
    ModelConfig(name, model_type="sentence_transformer")
    for name in SENTENCE_TRANSORMER_MODELS
]
# MODELS = [ModelConfig("Geotrend/bert-base-25lang-cased", model_type="sentence_transformer")]


########################
# Step 2 : Setup tasks #
########################
TASK_LIST_CLASSIFICATION = [
    "AmazonReviewsClassification",
    "MasakhaNEWSClassification", # bug when evealuating with bloom, need to enable truncation
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

TASKS = TASK_LIST_CLASSIFICATION

##########################
# Step 3 : Run benchmark #
##########################


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="fr")
    parser.add_argument("--batchsize", type=int, default=32)
    args = parser.parse_args()

    return args


def main(args):
    for model_config in MODELS:
        for task in TASKS:
            # change the task in the model config ! This is important to specify the chromaDB collection !
            model_name = model_config.model_name
            model_config.batch_size = args.batchsize
            print("Running task: ", task, "with model", model_name)
            eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
            evaluation = MTEB(tasks=[task], task_langs=[args.lang])
            evaluation.run(
                model_config,
                output_folder=f"results/{model_name}",
                batch_size=args.batchsize,
                eval_splits=eval_splits,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
