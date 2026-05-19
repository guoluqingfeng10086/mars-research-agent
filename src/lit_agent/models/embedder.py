# src/lit_agent/models/embedder.py

import torch
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查 PyTorch GPU 版本和显卡环境。")

        self.model_path = model_path
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = max_seq_length

    def encode_query(self, query: str):
        emb = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")[0]