import argparse
import logging
import json
import os
import subprocess
import time

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer 


MODELS = [
    "LASER2",
    "/gpfswork/rech/six/commun/models/sentence-transformers_average_word_embeddings_komninos",
    "/gpfswork/rech/six/commun/models/sentence-transformers_average_word_embeddings_glove.6B.300d",
    "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-nli-bitfit",
    "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-msmarco-specb-bitfit",
    "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-5.8B-weightedmean-nli-bitfit",
    "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    "/gpfswork/rech/six/commun/models/bigscience_sgpt-bloom-7b1-msmarco",
    "/gpfswork/rech/six/commun/models/bigscience-catalogue-lm-data_sgpt-bloom-1b3-nli",
    "/gpfswork/rech/six/commun/models/sentence-transformers_all-MiniLM-L6-v2",
    "/gpfswork/rech/six/commun/models/sentence-transformers_all-mpnet-base-v2",
    "/gpfswork/rech/six/commun/models/sentence-transformers_paraphrase-multilingual-mpnet-base-v2",
    "/gpfswork/rech/six/commun/models/sentence-transformers_sentence-t5-base",
    "/gpfswork/rech/six/commun/models/sentence-transformers_sentence-t5-xxl",
    "/gpfswork/rech/six/commun/models/sentence-transformers_gtr-t5-base",
    "/gpfswork/rech/six/commun/models/sentence-transformers_gtr-t5-xxl",
    "/gpfswork/rech/six/commun/models/nthakur_contriever-base-msmarco",
    "/gpfswork/rech/six/commun/models/sentence-transformers_msmarco-bert-co-condensor",
    "/gpfswork/rech/six/commun/models/bert-base-uncased",
    "/gpfswork/rech/six/commun/models/princeton-nlp_sup-simcse-bert-base-uncased",
    "/gpfswork/rech/six/commun/models/princeton-nlp_unsup-simcse-bert-base-uncased",
    "/gpfswork/rech/six/commun/models/sentence-transformers_LaBSE",
]

MODELS = [
    "/gpfswork/rech/six/commun/models/sentence-transformers_all-MiniLM-L12-v2",
    "/gpfswork/rech/six/commun/models/sentence-transformers_allenai-specter",
]

TASKS = [
    "STS15",
]

class SentenceTransformerSpecb(SentenceTransformer):
    # Requires: 
    # https://github.com/Muennighoff/sentence-transformers/tree/sgpt_poolings_specb
    # pip install git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokens = ["[SOS]", "{SOS}"]
        self._first_module().tokenizer.add_tokens(tokens, special_tokens=True)
        self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        self._first_module().bos_spec_token_q = self._first_module().tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d = self._first_module().tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_q_rep = self._first_module().tokenizer.encode("[", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_q = self._first_module().tokenizer.encode("]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d_rep = self._first_module().tokenizer.encode("{", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_d = self._first_module().tokenizer.encode("}", add_special_tokens=False)[0]
        self._first_module().replace_bos = True

    def encode(self, sentences, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # Add specb query token
        sentences = ["[SOS]" + sent for sent in sentences]
        return super().encode(sentences, **kwargs)

class SimCSEWrapper:
    def __init__(self, modelpath="princeton-nlp/sup-simcse-bert-base-uncased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.model = AutoModel.from_pretrained(modelpath).to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            # Get the embeddings
            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            all_embeddings.extend(embeddings.cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings

class LASER():
    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if os.path.exists("tmp.txt"):
            os.remove("tmp.txt")
        if os.path.exists("tmp.bin"):
            os.remove("tmp.bin")

        # LASER expects one text per line, so we need to replace newlines
        sentences = [s.replace("\n", " ") for s in sentences]
        with open("tmp.txt", "w") as f:
            f.write("\n".join(sentences))

        rc = subprocess.call("/gpfsscratch/rech/six/commun/commun/experiments/muennighoff/mteb/LASER/LASER_script.sh", shell=True)
              
        dim = 1024
        X = np.fromfile("tmp.bin", dtype=np.float32, count=-1)                                                                          
        X.resize(X.shape[0] // dim, dim)
        print(X.shape)
        return X


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--batchsize", type=int, default=32)
    args = parser.parse_args()
    return args

def main(args):

    out = {}
    for model_name in MODELS:
        if ("sgpt" in model_name.lower()) and ("msmarco" in model_name.lower()):
            model = SentenceTransformerSpecb(model_name) # Only used for SGPT-msmarco models
        elif "simcse" in model_name.lower():
            model = SimCSEWrapper(model_name)
        elif "LASER2" == model_name:
            model = LASER()
        else:
            model = SentenceTransformer(model_name)

        evaluation = MTEB(tasks=TASKS, task_langs=[args.lang])
        model_name = model_name.split("/")[-1].split("_")[-1]
        for task, task_name in zip(evaluation.tasks, TASKS):
            task.load_data()

            # Encode all with the same batch size for a fair comparison of speed / sentence
            data = task.dataset["test"]["sentence1"] + task.dataset["test"]["sentence2"]
            data_len = len(data)
            # Warmup run to build py caches etc
            embeddings = np.asarray(model.encode(data, batch_size=args.batchsize))
            tick = time.time()
            embeddings = np.asarray(model.encode(data, batch_size=args.batchsize))
            tock = time.time()

            out.setdefault(model_name, {})
            out[model_name].setdefault(task_name, {})
            out[model_name][task_name]["speed_ms"] = ((tock - tick) / data_len) * 1000
            out[model_name][task_name]["embedding_size_kb"] = embeddings.nbytes / data_len / 1000

        # Overwrite every iteration for intermed results
        with open("benchmark.json", "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
