import sys
import argparse
import logging

from mteb import MTEB

from src.ModelConfig import ModelConfig
from utils.tasks_list import get_tasks

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
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
    "intfloat/e5-mistral-7b-instruct",
    "Wissam42/sentence-croissant-llm-base"
]

# these models max_length is indicated to be 514 whereas the embedding layer actually supports 512
SENTENCE_TRANSORMER_MODELS_WITH_ERRORS = [
    "camembert/camembert-base",
    "camembert/camembert-large",
    "dangvantuan/sentence-camembert-large",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

UNIVERSAL_SENTENCE_ENCODER_MODELS = [
    "vprelovac/universal-sentence-encoder-multilingual-3",
    "vprelovac/universal-sentence-encoder-multilingual-large-3",
]

LASER_MODELS = ["laser2"]

VOYAGE_MODELS = ["voyage-2", "voyage-code-2"]

OPEN_AI_MODELS = ["text-embedding-ada-002"]

COHERE_MODELS = ["embed-multilingual-light-v3.0", "embed-multilingual-v3.0"]

MISTRAL_MODELS = ["mistral-embed"]

TYPES_TO_MODELS = {
    "sentence_transformer": SENTENCE_TRANSORMER_MODELS
    + SENTENCE_TRANSORMER_MODELS_WITH_ERRORS,
    "universal_sentence_encoder": UNIVERSAL_SENTENCE_ENCODER_MODELS,
    "laser": LASER_MODELS,
    "voyage_ai": VOYAGE_MODELS,
    "open_ai": OPEN_AI_MODELS,
    "cohere": COHERE_MODELS,
    "mistral_ai": MISTRAL_MODELS,

}

##########################
# Step 3 : Run benchmark #
##########################


def run_bitext_mining_tasks(args, model_config: ModelConfig, task: str):
    """Runs Bitext Mining tasks"""
    model_name = model_config.model_name
    model_config.batch_size = args.batchsize

    eval_splits = ["dev"] if task == "FloresBitextMining" else ["test"]

    if task == "DiaBLaBitextMining":
        evaluation = MTEB(tasks=[task], task_langs=[args.lang, "en"])
        evaluation.run(
            model_config,
            output_folder=f"results/{model_name}",
            batch_size=args.batchsize,
            eval_splits=eval_splits,
        )
    elif task == "FloresBitextMining":
        evaluation = MTEB(tasks=[task], task_langs=[args.lang, args.other_lang])
        evaluation.run(
            model_config,
            output_folder=f"results/{model_name}",
            batch_size=args.batchsize,
            eval_splits=eval_splits,
        )


def get_models(model_name, model_type, max_token_length) -> list[ModelConfig]:
    """Returns ModelConfig of input model_name or all ModelConfig model_type's list of models"""
    if model_name:
        logging.info(f"Running benchmark with the following model: {model_name}")
        if len(model_type) > 1:
            raise Exception(
                "Only one model type needs to be specified when a model name is given."
            )

        model_type_value = model_type[0]
        available_models_for_type = TYPES_TO_MODELS[model_type_value]

        if model_name not in available_models_for_type:
            raise Exception(
                f"Model name not in {available_models_for_type}.\n\
                Please select a correct model name corresponding to your model type."
            )

        if max_token_length:
            return [
                ModelConfig(
                    model_name=model_name,
                    model_type=model_type_value,
                    max_token_length=max_token_length,
                )
            ]
        else:
            return [ModelConfig(model_name=model_name, model_type=model_type_value)]
    else:
        logging.info(f"Running benchmark with the following model types: {model_type}")
        if max_token_length:
            return [
                ModelConfig(
                    name, model_type=model_type, max_token_length=max_token_length
                )
                for model_type in model_type
                for name in TYPES_TO_MODELS[model_type]
            ]
        else:
            return [
                ModelConfig(name, model_type=model_type)
                for model_type in model_type
                for name in TYPES_TO_MODELS[model_type]
            ]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        (argparse.Namespace): the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="fr")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--model_type", nargs="+", default=["sentence_transformer"])
    parser.add_argument(
        "--model_name", type=str, default=None, help="Run evaluation on one model only."
    )
    parser.add_argument(
        "--task_type",
        nargs="+",
        default=["all"],
        help="Choose tasks to run the evaluation on.",
    )
    parser.add_argument(
        "--other_lang",
        type=str,
        default="en",
        help="Other language for Bitext Mining task",
    )
    parser.add_argument("--max_token_length", type=int, default=None)
    args = parser.parse_args()

    return args


def main(args):
    # Select tasks to run evaluation on, default is set to all tasks
    tasks = get_tasks(args.task_type)

    # Running one model at a time or all models
    models = get_models(
        model_name=args.model_name,
        model_type=args.model_type,
        max_token_length=args.max_token_length,
    )

    # Running evaluation on all models for selected tasks
    for model_config in models:
        # fix the max_seq_length for some models with errors
        if model_config.model_name in SENTENCE_TRANSORMER_MODELS_WITH_ERRORS:
            model_config.embedding_function.model._first_module().max_seq_length = 512

        for task_type, task in tasks:
            if (task_type == "bitextmining") or ("BitextMining" in task):
                logging.warning(
                    "If other_lang is not specified in args, then it is set to 'en' by default"
                )
                logging.info(
                    f"Running task: {task} with model {model_config.model_name}"
                )
                run_bitext_mining_tasks(args=args, model_config=model_config, task=task)
            else:
                # change the task in the model config ! This is important to specify the chromaDB collection !
                eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
                model_name = model_config.model_name
                model_config.batch_size = args.batchsize
                logging.info(f"Running task: {task} with model {model_name}")
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
