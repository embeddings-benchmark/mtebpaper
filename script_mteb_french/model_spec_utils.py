import tensorflow_text as text  # Required to load muse models
import tensorflow as tf
import tensorflow_hub as hub
import torch
import json

from sentence_transformers import SentenceTransformer
from transformers import BloomModel, AutoTokenizer
from laser_encoders import LaserEncoderPipeline


def count_torch_model_params(model, all_params=True):
    if all_params:
        num_params = sum(p.numel() for p in model.parameters())
    else:
        # trainable params
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def get_sentence_transformers_model_specs(model_path, all_params=True):
    model = SentenceTransformer(model_path)
    num_params = count_torch_model_params(model, all_params)
    embedding_size = model.encode("dummy sentence").shape[0]
    input_size = model._modules.get("0").max_seq_length
    return num_params, embedding_size, input_size


def get_udever_bloom_model_specs(model_path, all_params=True):
    model = BloomModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    boq, eoq, bod, eod = "[BOQ]", "[EOQ]", "[BOD]", "[EOD]"
    eoq_id, eod_id = tokenizer.convert_tokens_to_ids([eoq, eod])
    if tokenizer.padding_side != "left":
        print("!!!", tokenizer.padding_side)
        tokenizer.padding_side = "left"

    def encode(texts: list, is_query: bool = True, max_length=300):
        bos = boq if is_query else bod
        eos_id = eoq_id if is_query else eod_id
        texts = [bos + t for t in texts]
        encoding = tokenizer(
            texts, truncation=True, max_length=max_length - 1, padding=True
        )
        for ids, mask in zip(encoding["input_ids"], encoding["attention_mask"]):
            ids.append(eos_id)
            mask.append(1)
        inputs = tokenizer.pad(encoding, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state[:, -1]
        return embeds

    num_params = count_torch_model_params(model, all_params)
    embedding_size = encode(["dummy sentence"]).shape[1]
    input_size = model.config.seq_length
    return num_params, embedding_size, input_size


def get_tfhub_model_specs(model_path, all_params=True):
    with open("universal_sentence_encoder_models_paths.json", "r") as f:
        models_paths = json.load(f)
    model_path = models_paths[model_path]
    model_keras = hub.KerasLayer(model_path)
    all_weights = model_keras.weights if all_params else model_keras.trainable_variables
    num_params = int(sum(tf.size(p).numpy() for p in all_weights))
    model = hub.load(model_path)

    def embed(input):
        return model(input)

    embedding_size = int(embed("dummy sentence").shape[1])
    input_size = None
    return num_params, embedding_size, input_size


def get_laser_model_specs(model_path, all_params=True):
    encoder = LaserEncoderPipeline(lang="fra_Latn")
    num_params = count_torch_model_params(encoder.encoder.encoder, all_params)
    embedding_size = encoder.encode_sentences(["dummy sentence"]).shape[1]
    input_size = encoder.encoder.max_tokens
    return num_params, embedding_size, input_size
