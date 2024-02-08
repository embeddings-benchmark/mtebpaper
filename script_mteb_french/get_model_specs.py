import json
from typing import Dict
import sys
import os

from huggingface_hub import HfFileSystem
import model_spec_utils
from run_benchmark import TYPES_TO_MODELS
import pandas as pd

"""
This script's purpose is to get the specifications of models in terms of :
- Computed size with parameters as float32 (in GB)
- Model files size (in GB)
- Number of parameters
- Input size (number of tokens)
- Embedding size (vector dimension)
"""

GET_SPEC_FUNCTIONS = {
    "sentence_transformer": "get_sentence_transformers_model_specs",
    "udever_bloom": "get_udever_bloom_model_specs",
    "universal_sentence_encoder": "get_tfhub_model_specs",
    "laser": "get_laser_model_specs",
}
ADD_TO_CHARACTERISTICS_CSV = True

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

    num_params, embedding_size, input_size = getattr(
        model_spec_utils, GET_SPEC_FUNCTIONS[model_type]
    )(model_name)

    model_size_float32_in_gb = num_params * 4 / 1e9
    model_size_with_hf_filesystem = (
        get_size_using_HF_filsystem(model_name)
        if model_type in ["sentence_transformers", "udever_bloom"]
        else None
    )
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


results_path = "results_analysis/model_specs.json"

if os.path.exists(results_path):
    with open(results_path) as f:
        SPECS = json.load(f)
else:
    SPECS = {}

for model_type in ["sentence_transformer", "universal_sentence_encoder", "laser"]:
    for model_name in TYPES_TO_MODELS[model_type]:
        print(f"Getting specs for {model_name}")
        if model_name in SPECS:
            print("Already computed")
        else:
            SPECS[model_name] = get_model_specs(model_name, model_type)
            with open(results_path, "w") as f:
                json.dump(SPECS, f, indent=4)

if ADD_TO_CHARACTERISTICS_CSV:
    computed_specs_df = (
        pd.DataFrame.from_records(SPECS)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "model",
                "computed_size_in_gb": "size_gb2",
                "n_params": "number_params2",
                "input_size": "seq_len2",
                "embedding_size": "embedding_dim2",
            }
        )[["model", "number_params2", "size_gb2", "seq_len2", "embedding_dim2"]]
    )
    base_file_df = pd.read_csv("results_analysis/models_characteristics.csv")
    merged_df = base_file_df.merge(computed_specs_df, on="model", how="left")
    merged_df["number_params"] = merged_df["number_params2"].fillna(
        merged_df["number_params"]
    )
    merged_df["size_gb"] = merged_df["size_gb2"].fillna(merged_df["size_gb"])
    merged_df["seq_len"] = merged_df["seq_len2"].fillna(merged_df["seq_len"])
    merged_df["embedding_dim"] = merged_df["embedding_dim2"].fillna(
        merged_df["embedding_dim"]
    )
    merged_df = merged_df.drop(
        columns=["number_params2", "size_gb2", "seq_len2", "embedding_dim2"]
    )
    merged_df.to_csv("results_analysis/models_characteristics.csv", index=False)
