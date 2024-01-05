import torch
from sentence_transformers import SentenceTransformer
from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

"""
IMPORTANT: This script is used to override this :
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

as the embedding function provided by chroma generates bug for not native sentence_transformer models
"""


class SentenceTransformerEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "dangvantuan/sentence-camembert-base",
        max_token_length: int = 4096,
        normalize_embeddings=True,
    ):
        super().__init__(max_token_length)

        self._model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input, normalize_embeddings=self.normalize_embeddings
        )

        return embeddings.tolist()
