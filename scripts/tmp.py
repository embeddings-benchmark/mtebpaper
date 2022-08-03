from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"

model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=["RedditClusteringP2P"])
evaluation.run(model, output_folder=f"results/{model_name}")
