import tiktoken
import torch
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings

"""
IMPORTANT: This script is used to override this :
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

as the embedding function provided by chroma generates bug for not native sentence_transformer models
"""
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self,
            model_name:str="dangvantuan/sentence-camembert-base",
            max_token_length:int=4096,
            normalize_embeddings=True
            ):
        self._model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.max_token_length = max_token_length
        # Use tiktoken to compute token length
        # As we may not know the exact tokenizer used for the model, we generically use the one of adav2
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
            )


    def truncate_sentences(self, sentences:Documents) -> Documents:
        """Truncates the sentences considering the max context window of the model

        Args:
            sentences (Documents): a list a sentences (documents)

        Returns:
            Documents: the truncated documents
        """
        for i, s in enumerate(sentences):
            tokenized_string = self.tokenizer.encode(s)
            # if string too large, truncate, decode, and replace
            if len(tokenized_string) > self.max_token_length:
                tokenized_string = tokenized_string[:self.max_token_length]
                sentences[i] = self.tokenizer.decode(tokenized_string)
        return sentences


    def __call__(self, input: Documents) -> Embeddings:
        input = list(input)
        input = self.truncate_sentences(input)
        embeddings = self.model.encode(input, normalize_embeddings=self.normalize_embeddings)
        embeddings = embeddings.tolist()
        return embeddings

