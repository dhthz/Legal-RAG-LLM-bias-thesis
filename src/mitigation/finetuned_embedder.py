from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

from src.mitigation.base import Corpus, MitigationArm

# Arm 4: the embedder fine-tuned with counterfactual data augmentation (label-preserving CDA: each training query and
# its he/she twin share the same relevant chunk), trained and selected with the recipe of Kim, Springer, Raghunathan &
# Sap (ACL Findings 2025, "Mitigating Bias in RAG: Controlling the Embedder"). ft_control is the same fine-tune
# without the twins. The corpus is re-embedded with each model and the query is embedded by the same model; chunk
# texts and metadata are the frozen ones. Built by scripts/chunking_and_embeddings/finetune_cda.py.
# Arm 4b, ft_ccd: the embedder fine-tuned with content conditional debiasing (Deng et al. 2024), built by
# scripts/chunking_and_embeddings/finetune_ccd.py.

MODELS = {"cda": ("models/cda", "faiss_indices/cda_paragraph_chunks_l2.index"),
          "ft_control": ("models/ft_control", "faiss_indices/ft_control_paragraph_chunks_l2.index"),
          "ft_ccd": ("models/ft_ccd", "faiss_indices/ft_ccd_paragraph_chunks_l2.index")}


def load_encoder(path: str, device: str = "cuda") -> SentenceTransformer:
    model = SentenceTransformer(path, trust_remote_code=True, device=device)
    # Same max-length warm-up as ChunkRetriever, which pins nomic's stateful rotary scaling
    model.encode(["legal " * (model.max_seq_length + 1000)], normalize_embeddings=True)
    return model


class FineTunedEmbedder(MitigationArm):
    name = "cda"

    def __init__(self, model_path: Optional[str] = None, index_path: Optional[str] = None):
        default_model, default_index = MODELS[self.name]
        self._corpus = Corpus(faiss.read_index(index_path or default_index),
                              encoder=load_encoder(model_path or default_model))

    def corpus(self) -> Corpus:
        return self._corpus


class FineTunedControl(FineTunedEmbedder):
    name = "ft_control"


class FineTunedCCD(FineTunedEmbedder):
    name = "ft_ccd"
