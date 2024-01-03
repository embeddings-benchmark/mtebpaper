import os
from chromadb import  Documents, Embeddings
from dotenv import load_dotenv
import voyageai as vai

from .CustomEmbeddingFunction import CustomEmbeddingFunction

# load the API key from .env
load_dotenv()

class VoyageAIEmbeddingFunction(CustomEmbeddingFunction):
    def __init__(self,
            model_name:str="voyage-lite-01",
            max_token_length:int=4096,
            ):
        super().__init__(max_token_length)
        
        self._model_name = model_name

        vai.api_key = os.environ.get("VOYAGE_API_KEY", None)


    def encode_sentences(self, input:Documents) -> Embeddings: 
        return vai.get_embeddings(input, model=self._model_name, input_type=None)


