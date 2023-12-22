from typing import List
import chromadb
from chromadb import EmbeddingFunction

class ChromaDBEmbedder:
    """Handles the mecanics of producing and saving embeddings in chromaDB
    It needs an embedding function, as described in the chromaDB's documentation
    """
    def __init__(
            self,
            embedding_function:EmbeddingFunction=None,
            task_name:str="default_collection",
            batch_size:int=32,
            save_embbedings:bool=True,
            path_to_chromadb="./chromaDB",
            **kwargs,
        ):
        self.client = chromadb.PersistentClient(path=path_to_chromadb)

        self.batch_size = batch_size
        self.save_embbeddings = save_embbedings
        if embedding_function is None:
            raise ValueError(f"You must provide an embedding function. Embedding functions available are {chromadb.utils.embedding_functions.get_builtins()}. For more information, please visit : https://docs.trychroma.com/embeddings")
        else:
            self.embedding_function = embedding_function
        self._task_name = task_name
        # setup the chromaDB collection
        self.set_collection(self.task_name)


    def encode(self, sentences:List[str], **kwargs):

        # if sentences is a string, change to List[str]
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i : i + self.batch_size]
            # check if we have the embedding in chroma
            sentences_in_chroma = self.collection.get(ids=batch_sentences, include=["documents", "embeddings"])
            # if we already have all the sentences in chroma, add those embeddings those to return
            if len(batch_sentences) == len(sentences_in_chroma["documents"]):
                all_embeddings.extend(sentences_in_chroma["embeddings"])
            # if we don't have all the sentences in chroma...
            else:
                missing_sentences = [s for s in batch_sentences if s not in sentences_in_chroma["ids"]]
                if self.save_embbeddings:
                    # We use the sentence as its own id in the database : not very clean, simplifies retrieving the sentence later
                    self.collection.upsert(
                        documents=missing_sentences,
                        ids=missing_sentences
                        )
                    all_embeddings.extend(self.collection.get(ids=batch_sentences, include=["embeddings"])["embeddings"])
                else:
                    all_embeddings.extend(self.embedding_function(missing_sentences) + sentences_in_chroma["embeddings"])
                # we may need to wait to avoid throttling the api
                # time.sleep(1)
        return all_embeddings

  
    @property
    def task_name(self):
        return self._task_name


    @task_name.setter
    def task_name(self, value):
        # if attribute "task_name" changes, change the collection
        self.set_collection(value)
        self._task_name = value


    def set_collection(self, collection_name:str):
        """Set the collection. Used whenever self.task_name is changed

        Args:
            collection_name (str): the name of the collection
        """
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
            )