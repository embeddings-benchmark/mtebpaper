import os
os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
from mteb import MTEB
from sentence_transformers import SentenceTransformer
class SentenceTransformerSpecb(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokens = ["[SOS]", "{SOS}"]
        self._first_module().tokenizer.add_tokens(tokens, special_tokens=True)
        self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        self._first_module().bos_spec_token_q = self._first_module().tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d = self._first_module().tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_q_rep = self._first_module().tokenizer.encode("[", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_q = self._first_module().tokenizer.encode("]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d_rep = self._first_module().tokenizer.encode("{", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_d = self._first_module().tokenizer.encode("}", add_special_tokens=False)[0]
        self._first_module().replace_bos = True
    def encode(self, sentences, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # Add specb token
        sentences = ["[SOS]" + sent for sent in sentences]
        return super().encode(sentences, **kwargs)
def main():
    model_path = "/gpfswork/rech/six/commun/models/Muennighoff_SGPT-125M-weightedmean-msmarco-specb-bitfit"
    model_name = model_path.split("/")[-1].split("_")[-1]
    model = SentenceTransformerSpecb(model_path)
    evaluation = MTEB(tasks=["TRECCOVID"])
    evaluation.run(model, output_folder=f"../results/{model_name}")

if __name__ == '__main__':
    main()
