# src/lit_agent/data/corpus_store.py

import pickle
from typing import List, Dict, Any

import numpy as np


class CorpusStore:
    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path
        self.data = None
        self.records: List[Dict[str, Any]] = []

        self.research_embeddings = None
        self.abstract_embeddings = None
        self.has_research_content = None
        self.has_abstract_note = None

    def load(self):
        with open(self.pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.records = self.data["records"]

        self.research_embeddings = np.stack([
            r["research_content_embedding"] for r in self.records
        ]).astype("float32")

        self.abstract_embeddings = np.stack([
            r["abstract_note_embedding"] for r in self.records
        ]).astype("float32")

        self.has_research_content = np.array([
            bool(r.get("has_research_content", False)) for r in self.records
        ])

        self.has_abstract_note = np.array([
            bool(r.get("has_abstract_note", False)) for r in self.records
        ])

        return self

    def __len__(self):
        return len(self.records)

    def get_record(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    def basic_info(self):
        return {
            "num_records": len(self.records),
            "embedding_dim": self.research_embeddings.shape[1],
            "pkl_path": self.pkl_path,
            "model_path": self.data.get("model_path", ""),
        }