import os
from dotenv import load_dotenv

from .ChromaDBEmbedder import ChromaDBEmbedder
from .CohereEmbeddingFunction import CohereEmbeddingFunction
from .LaserEmbeddingFunction import LaserEmbeddingFunction
from .MistralAIEmbeddingFunction import MistralAIEmbeddingFunction
from .OpenAIEmbeddingFunction import OpenAIEmbeddingFunction
from .SentenceTransformerEmbeddingFunction import SentenceTransformerEmbeddingFunction
from .UniversalSentenceEncoderEmbeddingFunction import (
    UniversalSentenceEncoderEmbeddingFunction,
)
from .VoyageAIEmbeddingFunction import VoyageAIEmbeddingFunction

load_dotenv()


class ModelConfig(ChromaDBEmbedder):
    """simple class to get the model name and type"""

    def __init__(
        self,
        model_name: str,
        model_type: str = None,
        max_token_length: int = None,
    ):
        """The model configuration to use for the benchmark

        Args:
            model_name (str): The name of the model
                for example "voyage-lite-01", "text-embedding-ada-002", or "flaubert/flaubert_base_uncased"
            model_type (str, optional): the type of the model. Defaults to None.
                for example: "voyage_ai", "open_ai" or "sentence_transformer"
            max_token_length (int, optional): the maximum length of the context window. Defaults to None.
                if None, it will be set to the max size specified by the model provided
            task_name (str, optional): the name of the task. Used in the ChromaDBEmbedder to specify
                in which collection to save the data to encode
        """
        self._available_model_types = list(self._max_token_per_model.keys())

        self.model_name = model_name
        self._model_type = (
            model_type
            if model_type in self._max_token_per_model.keys()
            else self.infer_model_type(model_name)
        )
        self._max_token_length = max_token_length
        self.embedding_function = self.get_embedding_function()

        # inherit the saving of embeddings, and encoding logic from ChromDBEmbedder
        save_embbeddings = (
            False
            if (
                model_type == "sentence_transformer"
                or model_type == "universal_sentence_encoder"
            )
            else True
        )
        super().__init__(
            self.embedding_function,
            save_embbedings=save_embbeddings,
        )

    @property
    def _max_token_per_model(self):
        return {
            "cohere": 4096,
            "laser": 512,
            "voyage_ai": 4000,
            "open_ai": 8191,
            "sentence_transformer": 4096,
            "universal_sentence_encoder": 4096,
            "mistral_ai": 4096,
        }

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        if value not in self._max_token_per_model.keys():
            raise ValueError(
                f"Please specify model type among supported types {self._available_model_types}"
            )
        self._model_type = value

    @property
    def max_token_length(self):
        true_max = self._max_token_per_model[self.model_type]
        if self._max_token_length is None or self._max_token_length >= true_max:
            return true_max

        return self._max_token_length

    @max_token_length.setter
    def max_token_length(self, value):
        true_max = self._max_token_per_model[self.model_type]
        if value >= true_max:
            raise ValueError(
                f"The max token length for model {self.model_name} is '{true_max}'"
            )
        self._max_token_length = value

    def infer_model_type(self, model_name):
        print(
            "The provided model type is not recognized. Trying to infer model type using model name..."
        )
        raise NotImplementedError(
            f"Please specify model type among supported types : {self._available_model_types}"
        )

    def get_embedding_function(self):
        match self.model_type:
            case "cohere":
                return CohereEmbeddingFunction(self.model_name, self.max_token_length)
            case "laser":
                return LaserEmbeddingFunction(self.model_name, self.max_token_length)
            case "voyage_ai":
                return VoyageAIEmbeddingFunction(self.model_name, self.max_token_length)
            case "open_ai":
                return OpenAIEmbeddingFunction(self.model_name, self.max_token_length)
            case "sentence_transformer":
                return SentenceTransformerEmbeddingFunction(
                    self.model_name, self.max_token_length
                )
            case "universal_sentence_encoder":
                return UniversalSentenceEncoderEmbeddingFunction(
                    self.model_name, self.max_token_length
                )
            case "mistral_ai":
                return MistralAIEmbeddingFunction(
                    self.model_name, self.max_token_length
                )