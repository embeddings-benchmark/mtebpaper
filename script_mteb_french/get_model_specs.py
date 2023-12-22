import sys
import json
from huggingface_hub import HfFileSystem
from sentence_transformers import SentenceTransformer


SENTENCE_TRANSFORMER_MODELS = [
    "bert-base-multilingual-cased",
]

HFFS = HfFileSystem()


def get_model_specs(model_name):
        
    model = SentenceTransformer(model_name)
    specs = {
        "size_in_GB": get_size_using_HF_filsystem(model_name),
        "n_params": model._parameters,
        "input_size": model._modules.get("0").max_seq_length,
        "embedding_size": model.encode("dummy sentence").shape[0]
    }
    return specs


def get_size_of_object(model, seen=None):
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
    elif hasattr(model, '__dict__'):
        size += get_size_of_object(model.__dict__, seen)
    elif hasattr(model, '__iter__') and not isinstance(model, (str, bytes, bytearray)):
        size += sum([get_size_of_object(i, seen) for i in model])
    return round(size/1000000, 3)


def get_size_using_HF_filsystem(model_name:str):
    """Uses the huggingface filesystem
    to find the model_pytorch.bin

    Args:
        model_name (str): the name of the model on Hf
    """
    files = HFFS.ls(model_name, detail=True)

    return next((round(f["size"]/1e9,3) for f in files
        if f["name"] == f"{model_name}/pytorch_model.bin"), None)


SPECS = {}
for model_name in SENTENCE_TRANSFORMER_MODELS:
    SPECS[model_name] = get_model_specs(model_name)

with open("model_specs.json", "w") as f:
    json.dump(SPECS, f)
