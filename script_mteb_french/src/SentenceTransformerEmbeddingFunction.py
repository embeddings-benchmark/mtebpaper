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
        prompts: dict = None,
        task_type: str = None,
    ):
        super().__init__(max_token_length)

        self._model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.prompts = prompts
        self.task_type = task_type # if provided, used for prompt selection

        if (isinstance(prompts, dict)) and (task_type is not None) and (task_type not in self.prompts.keys()):
            raise ValueError(f"Please provide task type from the following list: {self.prompts.keys()}")

        self.model = SentenceTransformer(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu",
            prompts=prompts, 
            default_prompt_name=None, # Set to None for classical use cases with no prompts
        )

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        if (self.task_type is not None) and (self.task_type in self.model.prompts.keys()):
            prompt_name = self.task_type
            embeddings = self.model.encode(
                input, normalize_embeddings=self.normalize_embeddings,
                prompt_name=prompt_name
            )
        else:
            embeddings = self.model.encode(
                input, normalize_embeddings=self.normalize_embeddings,
            )

        return embeddings.tolist()
