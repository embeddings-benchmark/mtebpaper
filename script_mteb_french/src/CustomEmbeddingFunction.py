import tiktoken
from chromadb import EmbeddingFunction, Documents, Embeddings


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self,
            max_token_length:int=4096,
            ):
        self.max_token_length = max_token_length
        # Use tiktoken to compute token length
        # As we may not know the exact tokenizer used for the model, we generically use the one of adav2
        self.tokenizer = tiktoken.get_encoding("cl100k_base")


    def truncate_sentences(self, sentences:Documents) -> Documents:
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
        input = self.truncate_sentences(input)
        embeddings = self.encode_sentences(input)

        return embeddings
    
    
    def encode_sentences(self, input: Documents) -> Embeddings:
        """Needs to be implemented by the sub class
        """

        raise NotImplementedError()

