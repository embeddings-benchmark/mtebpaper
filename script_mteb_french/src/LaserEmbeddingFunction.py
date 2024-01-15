import os

from chromadb import Documents, Embeddings
from laser_encoders import LaserEncoderPipeline

from .AbstractEmbeddingFunction import AbstractEmbeddingFunction


class LaserEmbeddingFunction(AbstractEmbeddingFunction):
    def __init__(
        self,
        model_name: str,
        max_token_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        super().__init__(max_token_length)
        self._model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self._download_laser_models()

        self.encoder = LaserEncoderPipeline(lang="fra_Latn")

    @property
    def model_name(self):
        return self._model_name

    def encode_documents(self, input: Documents) -> Embeddings:
        embeddings = self.encoder.encode_sentences(
            input, normalize_embeddings=self.normalize_embeddings
        )
        return embeddings.tolist()

    @staticmethod
    def _download_laser_models():
        MODELS_DOWNLOAD_FOLDER = "models"
        if not os.path.exists(MODELS_DOWNLOAD_FOLDER):
            os.mkdir(MODELS_DOWNLOAD_FOLDER)

        # This instruction will by default install laser2
        os.system(
            'python -m laser_encoders.download_models --lang="fra_Latn" --model-dir="{0}"'.format(
                MODELS_DOWNLOAD_FOLDER
            )
        )
