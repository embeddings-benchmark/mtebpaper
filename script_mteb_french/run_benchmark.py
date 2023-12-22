import argparse

from mteb import MTEB

from src.ModelConfig import ModelConfig

# Build list of model configs to use for benchmark
## ModelConfig(model_name, model_type, max_token_length(optional))
MODELS = [
    ModelConfig("voyage-lite-01", model_type="voyage_ai")
    ]

TASKS = [
    "SyntecRetrieval"
]


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
            model_config.task_name = task
            model_name = model_config.model_name
            print("Running task: ", task, "with model", model_name)
            eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
            evaluation = MTEB(tasks=[task], task_langs=[args.lang])
            evaluation.run(
                model_config, output_folder=f"results/{model_name}",
                batch_size=args.batchsize,
                eval_splits=eval_splits,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)