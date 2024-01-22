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
    "sentence-transformers/LaBSE",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-small",
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
    "izhx/udever-bloom-3b", # too big
    "izhx/udever-bloom-7b1", # too big
    "intfloat/e5-mistral-7b-instruct", # too big
]
"""

# these models max_length is indicated to be 514 whereas the embedding layer actually supports 512
SENTENCE_TRANSORMER_MODELS_WITH_ERRORS = [
    "camembert/camembert-base",
    "camembert/camembert-large",
    "dangvantuan/sentence-camembert-large",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

UNIVERSAL_SENTENCE_ENCODER_MODELS = [
    "vprelovac/universal-sentence-encoder-multilingual-3"
]

LASER_MODELS = ["laser2"]

VOYAGE_MODELS = ["voyage-02"]

OPEN_AI_MODELS = ["text-embedding-ada-002"]

COHERE_MODELS = ["embed-multilingual-light-v3.0", "embed-multilingual-v3.0"]

TYPES_TO_MODELS = {
    "sentence_transformer": SENTENCE_TRANSORMER_MODELS
    + SENTENCE_TRANSORMER_MODELS_WITH_ERRORS,
    "universal_sentence_encoder": UNIVERSAL_SENTENCE_ENCODER_MODELS,
    "laser": LASER_MODELS,
    "voyage_ai": VOYAGE_MODELS,
    "open_ai": OPEN_AI_MODELS,
    "cohere": COHERE_MODELS,
}

########################
# Step 2 : Setup tasks #
########################
TASK_LIST_CLASSIFICATION = [
    "AmazonReviewsClassification",
    "MasakhaNEWSClassification",  # bug when evealuating with bloom, need to enable truncation
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
    # "FloresBitextMining",
]

TASKS = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_SUMMARIZATION
    + TASK_LIST_BITEXTMINING
)

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
    parser.add_argument("--model_type", nargs="+", default=["sentence_transformer"])
    args = parser.parse_args()

    return args


def main(args):
    print("Running benchmark with the following model types: ", args.model_type)
    models = [
        ModelConfig(name, model_type=model_type)
        for model_type in args.model_type
        for name in TYPES_TO_MODELS[model_type]
    ]
    for model_config in models:
        # fix the max_seq_length for some models with errors
        if model_config.model_name in SENTENCE_TRANSORMER_MODELS_WITH_ERRORS:
            model_config.embedding_function.model._first_module().max_seq_length = 512
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
