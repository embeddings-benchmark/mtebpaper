import argparse

from mteb import MTEB

from src.ModelConfig import ModelConfig


# if False, default sample models will be used, instead of those specified in the lists
# will be removed when everything is ready
benchmark_is_ready = False
#############################
# Step 1 : Setup model list #
#############################
SENTENCE_TRANSORMER_MODELS = [
    #"camembert/camembert-base",
    #"camembert/camembert-large",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "flaubert/flaubert_base_uncased",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    "dangvantuan/sentence-camembert-base",
    "dangvantuan/sentence-camembert-large",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-large",
    "woody72/multilingual-e5-base",
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
    "izhx/udever-bloom-3b",
    "izhx/udever-bloom-7b1"
]


LASER_MODELS = ["Laser2"]

VOYAGE_MODELS = ["voyage-lite-01", "voyage-01"]

OPEN_AI_MODELS = ["text-embedding-ada-002"]

########################
# Step 2 : Setup tasks #
########################
TASKS = ["SyntecRetrieval", "AlloprofRetrieval"]

##########################
# Step 3 : Run benchmark #
##########################

# Build list of model configs based on the lists above
## ModelConfig(model_name, model_type, max_token_length(optional))
if benchmark_is_ready:
    MODELS = [
        ModelConfig(name, model_type="sentence_transformer")
        for name in SENTENCE_TRANSORMER_MODELS
    ]
    MODELS.extend([ModelConfig(name, model_type="voyage_ai") for name in VOYAGE_MODELS])
    MODELS.extend([ModelConfig(name, model_type="open_ai") for name in OPEN_AI_MODELS])
    MODELS.extend([ModelConfig(name, model_type="laser") for name in LASER_MODELS])
else:
    MODELS = [ModelConfig(name, model_type="voyage_ai") for name in VOYAGE_MODELS]


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
