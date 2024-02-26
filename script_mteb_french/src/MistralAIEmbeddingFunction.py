import os
import time
from chromadb import Documents, Embeddings
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction


# load the API key from .env
load_dotenv()


class MistralAIEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "mistral-embed",
        max_token_length: int = 8191,
    ):
        self._model_name = model_name
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
        try:
            embeddings_batch_response = self.client.embeddings(
                model=self.model_name,
                input=input,
            )
            time.sleep(.5) # rate limit of 2 requests per seconds
            return [emb.embedding for emb in embeddings_batch_response.data]
        
        except MistralAPIException as e:
            if "Too many tokens in batch." in e.message:
                subbatches = self.split_big_batches(input)
                embeddings = []
                for batch in subbatches:
                    embeddings_batch_response = self.client.embeddings(
                                    model=self.model_name,
                                    input=batch,
                                )
                    time.sleep(.5) # rate limit of 2 requests per seconds
                    embeddings.extend([emb.embedding for emb in embeddings_batch_response.data])
                return embeddings
            else:
                raise MistralAPIException(e)

    def split_big_batches(self, input: Documents, batch_token_limit=14000) -> list[Documents]:
        """The max token limit of a single batch api cal for mistral ai is 16384 tokens.
        This functions splits batches that are to big into smaller subatches

        Args:
            input (Documents): a batch of documents
            batch_token_limit (int): the api's token limit for a single batch

        Returns:
            list[Documents]: a list of batches (list of list)
        """
        token_per_document = [len(self.tokenizer.encode(d)) for d in input]
        subatches = []
        curr_batch = []
        curr_batch_size = 0
        for doc, doc_size in zip(input, token_per_document):
            if curr_batch_size + doc_size > batch_token_limit:
                subatches.append(curr_batch)
                curr_batch = []
                curr_batch_size = 0
            curr_batch.append(doc)
            curr_batch_size += doc_size
        if curr_batch:
            subatches.append(curr_batch)

        assert len(input) == len([d for subbatch in subatches for d in subbatch]),\
        f"Got {len(input)} documents in input, but only {len([d for subbatch in subatches for d in subbatch])} document in subbatches"

        return subatches