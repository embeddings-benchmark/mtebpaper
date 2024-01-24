import json
import tensorflow_text  # Required to load muse models
import tensorflow as tf
import tensorflow_hub as hub
from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction


class UniversalSentenceEncoderEmbeddingFunction(AbstractEmbeddingFunction):
    MODELS_PATHS = "script_mteb_french/universal_sentence_encoder_models_paths.json"

    def __init__(
        self,
        model_name: str = "vprelovac/universal-sentence-encoder-multilingual-3",
        max_token_length: int = 4096,
    ):
        super().__init__(max_token_length)

        self._model_name = model_name

        self.models_paths = self._load_model_paths()
        print(self.models_paths)

        if model_name in self.models_paths.keys():
            model_path = self.models_paths[model_name]
        else:
            raise KeyError(f"Model not found in list {self.models_paths.keys()}")
        
        self.model = hub.load(model_path)

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        return self.model(input).numpy()
    
    @classmethod
    def _load_model_paths(cls):
        return json.load(open(cls.MODELS_PATHS, "r"))
        