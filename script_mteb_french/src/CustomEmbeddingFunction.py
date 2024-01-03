import tiktoken
from chromadb import EmbeddingFunction, Documents, Embeddings
from abc import ABC, abstractmethod


class CustomEmbeddingFunction(EmbeddingFunction, ABC):
    def __init__(self,
            max_token_length:int=4096,
            ):
        self.max_token_length = max_token_length
        # Use tiktoken to compute token length
        # As we may not know the exact tokenizer used for the model, we generically use the one of adav2
        self.tokenizer = tiktoken.get_encoding("cl100k_base")


    def truncate_documents(self, sentences:Documents) -> Documents:
        """Truncates the sentences considering the max context window of the model

        Args:
            sentences (Documents): a list a sentences (documents)

        Returns:
            Documents: the truncated documents
        """
        for i, s in enumerate(sentences):
            tokenized_strings = self.tokenizer.encode(s)
            # if string too large, truncate, decode, and replace
            if len(tokenized_strings) > self.max_token_length:
                s = s[:self.max_token_length]
                sentences[i] = self.tokenizer.decode(s)
        return sentences


    def __call__(self, input: Documents) -> Embeddings:
        """Wrapper that truncates the documents, encodes them

        Args:
            input (Documents): List of documents

        Returns:
            Embeddings: the encoded sentences
        """ 
        input = self.truncate_documents(input)
        embeddings = self.encode_documents(input)

        return embeddings
    
    
    @abstractmethod
    def encode_documents(self, input: Documents) -> Embeddings:
        """Needs to be implemented by the child class. Takes a list of strings
        and returns the corresponding embedding

        Args:
            input (Documents): list of documents (strings)

        Raises:
            NotImplementedError: Needs to be implements by child class

        Returns:
            Embeddings: list of embeddings
        """

        raise NotImplementedError()

