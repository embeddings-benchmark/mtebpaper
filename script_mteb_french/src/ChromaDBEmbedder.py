import os
from typing import List
import chromadb
from chromadb import EmbeddingFunction


class ChromaDBEmbedder:
    """Handles the mecanics of producing and saving embeddings in chromaDB
    It needs an embedding function, as described in the chromaDB's documentation : https://docs.trychroma.com/embeddings
    This embedding function specifies the wey embeddings are obtained, from a model or api
    """

    def __init__(
        self,
        embedding_function: EmbeddingFunction = None,
        batch_size: int = 32,
        save_embbedings: bool = True,
        path_to_chromadb="./ChromaDB",
        **kwargs,
    ):
        self.batch_size = batch_size
        self.save_embbeddings = save_embbedings
        if embedding_function is None:
            raise ValueError(
                f"You must provide an embedding function. Embedding functions available are {chromadb.utils.embedding_functions.get_builtins()}. For more information, please visit : https://docs.trychroma.com/embeddings"
            )
        else:
            self.embedding_function = embedding_function

        # setup the chromaDB collection
        self.client = chromadb.PersistentClient(path=path_to_chromadb)
        collection_name = embedding_function.model_name.replace("/", "-")[:63]
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def encode(self, sentences: List[str], **kwargs):
        # if sentences is a string, change to List[str]
        if isinstance(sentences, str):
            sentences = [sentences]

        # use a dict to store a mapping of {sentence: embedding}
        # we have to do this because collection.get() returns embeddings in a random order...
        sent_emb_mapping = {}
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i : i + self.batch_size]
            # check if we have the embedding in chroma
            sentences_in_chroma = self.collection.get(
                ids=batch_sentences, include=["documents", "embeddings"]
            )

            # if we already have all the sentences in chroma, add those embeddings to the mapping dict
            if len(batch_sentences) == len(sentences_in_chroma["documents"]):
                sent_emb_mapping = sent_emb_mapping | dict(
                    zip(sentences_in_chroma["ids"], sentences_in_chroma["embeddings"])
                )

            # if we don't have all the sentences in chroma...
            else:
                missing_sentences = [
                    s for s in batch_sentences if s not in sentences_in_chroma["ids"]
                ]
                if self.save_embbeddings:
                    # We use the sentence as its own id in the database : not very clean, simplifies retrieving the sentence later
                    self.collection.upsert(
                        documents=missing_sentences, ids=missing_sentences
                    )
                    embs = self.collection.get(
                        ids=batch_sentences, include=["embeddings"]
                    )
                    sent_emb_mapping = sent_emb_mapping | dict(
                        zip(embs["ids"], embs["embeddings"])
                    )
                else:
                    # first add what we have in chroma
                    sent_emb_mapping = sent_emb_mapping | dict(
                        zip(
                            sentences_in_chroma["ids"],
                            sentences_in_chroma["embeddings"],
                        )
                    )
                    # then add what we obtain from encoding function
                    sent_emb_mapping = sent_emb_mapping | dict(
                        zip(
                            missing_sentences,
                            self.embedding_function(missing_sentences),
                        )
                    )
                # we may need to wait to avoid throttling the api
                # time.sleep(1)
        # return embeddings in correct order
        return [sent_emb_mapping[s] for s in sentences]
