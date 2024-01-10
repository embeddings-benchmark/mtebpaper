import os
from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import CohereEmbeddingFunction as CoEmbFunc
import cohere
from dotenv import load_dotenv

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

load_dotenv()


class CohereEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
            self, 
            model_name:str = "Cohere/Cohere-embed-multilingual-light-v3.0", 
            max_token_length: int = 512
        ):
        super().__init__(max_token_length)
        self._model_name = model_name

        api_key = os.getenv("COHERE_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Please make sure 'COHERE_API_KEY' is setup as an environment variable"
            )

        self.client = cohere.Client(api_key)
        CoEmbFunc.__init__(self, api_key=api_key, model_name=model_name)

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        return CoEmbFunc.__call__(self, input)
