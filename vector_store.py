"""
Lightweight vector store backed by numpy + JSON.
Drop-in replacement for ChromaDB with no heavy dependencies.
"""

import os
import json
import numpy as np

_STORE_DIR = "./vector_store"


class VectorStore:
    def __init__(self, path: str = _STORE_DIR):
        self.path = path
        self._ids:   list[str]        = []
        self._docs:  list[str]        = []
        self._metas: list[dict]       = []
        self._embs:  list[list[float]] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _meta_path(self) -> str:
        return os.path.join(self.path, "metadata.json")

    def _emb_path(self) -> str:
        return os.path.join(self.path, "embeddings.npy")

    def _load(self):
        if os.path.exists(self._meta_path()) and os.path.exists(self._emb_path()):
            with open(self._meta_path()) as f:
                data = json.load(f)
            self._ids   = data["ids"]
            self._docs  = data["docs"]
            self._metas = data["metas"]
            self._embs  = np.load(self._emb_path()).tolist()

    def _save(self):
        os.makedirs(self.path, exist_ok=True)
        with open(self._meta_path(), "w") as f:
            json.dump({"ids": self._ids, "docs": self._docs, "metas": self._metas}, f)
        np.save(self._emb_path(), np.array(self._embs, dtype=np.float32))

    # ── Public API ────────────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self._ids)

    def upsert(
        self,
        ids:        list[str],
        documents:  list[str],
        embeddings: list[list[float]],
        metadatas:  list[dict],
    ):
        idx_map = {id_: i for i, id_ in enumerate(self._ids)}
        for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            if id_ in idx_map:
                i = idx_map[id_]
                self._docs[i]  = doc
                self._embs[i]  = emb
                self._metas[i] = meta
            else:
                idx_map[id_] = len(self._ids)
                self._ids.append(id_)
                self._docs.append(doc)
                self._embs.append(emb)
                self._metas.append(meta)
        self._save()

    def query(self, query_embedding: list[float], n_results: int = 5) -> dict:
        if not self._embs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embedding, dtype=np.float32)
        E = np.array(self._embs, dtype=np.float32)

        q_norm = q / (np.linalg.norm(q) + 1e-10)
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
        sims   = E_norm @ q_norm

        k       = min(n_results, len(sims))
        top_idx = np.argsort(sims)[::-1][:k]

        return {
            "documents": [[self._docs[i]  for i in top_idx]],
            "metadatas": [[self._metas[i] for i in top_idx]],
            "distances": [[float(1 - sims[i]) for i in top_idx]],
        }
