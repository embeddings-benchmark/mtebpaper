import os
from chromadb import Documents, Embeddings
from dotenv import load_dotenv
from mistralai.client import MistralClient

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction


# load the API key from .env
load_dotenv()


class MistralAIEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "mistral-embed",
        max_token_length: int = 8191,
    ):
        AbstractEmbeddingFunction.__init__(self, max_token_length=max_token_length)

        api_key = os.environ.get("MISTRAL_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Please make sure 'MISTRAL_API_KEY' is setup as an environment variable"
            )
        
        self.client = MistralClient()


    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        embeddings_batch_response = self.client.embeddings(
            model=self.model,
            input=input,
        )

        return  [emb.embedding for emb in embeddings_batch_response.data]
