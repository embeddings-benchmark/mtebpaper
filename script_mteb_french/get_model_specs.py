import json
from typing import Dict
import sys

from huggingface_hub import HfFileSystem
import model_spec_utils

"""
This script's purpose is to get the specifications of models in terms of :
- Computed size with parameters as float32 (in GB)
- Model files size (in GB)
- Number of parameters
- Input size (number of tokens)
- Embedding size (vector dimension)

Just specify the names of the models for which you want the specs,
as lists (SENTENCE_TRANSFORMER_MODELS, UDEVER_BLOOM_MODELS, TFHUB_MODELS, 
LASER_MODELS)
"""

SENTENCE_TRANSFORMER_MODELS = [
    "bert-base-multilingual-cased",
    "dangvantuan/sentence-camembert-base",
]

UDEVER_BLOOM_MODELS = [
    'izhx/udever-bloom-560m'
]

TFHUB_MODELS = [
    'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
]

LASER_MODELS = [
    'french'
]

GET_SPEC_FUNCTIONS = {
    "sentence_transformers": "get_sentence_transformers_model_specs",
    "udever_bloom": "get_udever_bloom_model_specs",
    "tfhub": "get_tfhub_model_specs",
    "laser": "get_laser_model_specs"
}

HFFS = HfFileSystem()


def get_model_specs(model_name: str, model_type: str) -> Dict:
    """Get the specifications of a HF model

    Args:
        model_name (str): the hf name of the model
            (i.e. 'sentence-transformers/all-MiniLM-L6-v2')
        model_type (str): the type of model (sentence_transformers, udever_bloom, tfhub, laser)

    Returns:
        Dict: the specifications of the model
    """

    num_params, embedding_size, input_size = getattr(model_spec_utils, GET_SPEC_FUNCTIONS[model_type])(model_name)

    model_size_float32_in_gb = num_params * 4 / 1e9
    model_size_with_hf_filesystem = get_size_using_HF_filsystem(model_name) if model_type in ["sentence_transformers", "udever_bloom"] else None
    specs = {
        "computed_size_in_gb": model_size_float32_in_gb,
        "files_size_in_gb": model_size_with_hf_filesystem,
        "n_params": num_params,
        "input_size": input_size,
        "embedding_size": embedding_size,
    }
    return specs


def get_size_of_object(model, seen=None) -> float:
    """Recursively finds size of objects"""
    size = sys.getsizeof(model)
    if seen is None:
        seen = set()
    obj_id = id(model)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(model, dict):
        size += sum([get_size_of_object(v, seen) for v in model.values()])
        size += sum([get_size_of_object(k, seen) for k in model.keys()])
    elif hasattr(model, "__dict__"):
        size += get_size_of_object(model.__dict__, seen)
    elif hasattr(model, "__iter__") and not isinstance(model, (str, bytes, bytearray)):
        size += sum([get_size_of_object(i, seen) for i in model])
    return round(size / 1000000, 3)


def get_size_using_HF_filsystem(model_name: str):
    """Uses the huggingface filesystem
    to find the model_pytorch.bin

    Args:
        model_name (str): the name of the model on Hf
    """
    files = HFFS.ls(model_name, detail=True)

    return next(
        (
            round(f["size"] / 1e9, 3)
            for f in files
            if f["name"] == f"{model_name}/pytorch_model.bin"
        ),
        None,
    )


SPECS = {}
for model_name in SENTENCE_TRANSFORMER_MODELS:
    SPECS[model_name] = get_model_specs(model_name, "sentence_transformers")
for model_name in UDEVER_BLOOM_MODELS:
    SPECS[model_name] = get_model_specs(model_name, "udever_bloom")
for model_name in TFHUB_MODELS:
    SPECS[model_name] = get_model_specs(model_name, "tfhub")
for model_name in LASER_MODELS:
    SPECS[model_name] = get_model_specs(model_name, "laser")

with open("model_specs.json", "w") as f:
    json.dump(SPECS, f)
