import torch
from FlagEmbedding import BGEM3FlagModel
from chromadb import Documents, Embeddings

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction

"""
IMPORTANT: This script is used to override this :
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

as the embedding function provided by chroma generates bug for not native sentence_transformer models
"""


class FlagEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_token_length: int = 4096,
        normalize_embeddings=True,
        use_fp16=False,
    ):
        super().__init__(max_token_length)

        self._model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.model = BGEM3FlagModel(
            model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_fp16=use_fp16,
        )

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input,
            max_length=self.max_token_length,
            return_dense=True,
            return_sparse=False, 
            return_colbert_vecs=False
        )

        return embeddings['dense_vecs'].tolist()
