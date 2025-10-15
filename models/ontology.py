"""
Skill Ontology Normalizer for ResuMatch-X
- Groups related skills via MiniLM embeddings and rule-based clusters
- Outputs canonical categories and normalized skills JSON
"""
from __future__ import annotations

from typing import List, Dict, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SkillOntologyNormalizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedder: {model_name}")
        except Exception as e:
            logger.warning(f"MiniLM unavailable; ontology normalization will be rule-based only: {e}")
            self.embedder = None

        # Seeded rule-based clusters
        self.clusters = {
            "deep_learning_frameworks": {"aliases": ["tensorflow", "pytorch", "keras", "mxnet", "theano"]},
            "databases": {"aliases": ["mysql", "postgres", "postgresql", "oracle", "sqlite", "mongodb", "redis"]},
            "cloud": {"aliases": ["aws", "azure", "gcp", "google cloud", "amazon web services"]},
            "frontend": {"aliases": ["javascript", "js", "react", "angular", "vue", "html", "css", "bootstrap"]},
            "devops": {"aliases": ["docker", "kubernetes", "jenkins", "terraform", "ansible", "ci/cd"]},
            "ml_core": {"aliases": ["machine learning", "ml", "scikit-learn", "xgboost", "lightgbm"]},
            "nlp": {"aliases": ["nlp", "spacy", "transformers", "bert", "gpt", "llm"]},
            "data_eng": {"aliases": ["spark", "hadoop", "airflow", "kafka", "etl"]},
        }

    def _embed(self, items: List[str]) -> np.ndarray:
        if self.embedder is None:
            return np.zeros((len(items), 384), dtype=float)
        return np.asarray(self.embedder.encode(items))

    def normalize(self, skills: List[str]) -> Dict[str, Any]:
        skills = [s.strip().lower() for s in (skills or []) if isinstance(s, str) and s.strip()]
        skills = sorted(list(dict.fromkeys(skills)))
        if not skills:
            return {"normalized": {}, "mapping": {}, "unknown": []}

        # Rule-based mapping first
        mapping: Dict[str, str] = {}
        normalized: Dict[str, List[str]] = {k: [] for k in self.clusters.keys()}
        unknown: List[str] = []
        for sk in skills:
            placed = False
            for cname, spec in self.clusters.items():
                if sk in spec["aliases"]:
                    normalized[cname].append(sk)
                    mapping[sk] = cname
                    placed = True
                    break
            if not placed:
                unknown.append(sk)

        # If embedder available, assign unknown skills to closest cluster centroid
        if unknown and self.embedder is not None:
            cluster_names = list(self.clusters.keys())
            centroids = self._embed([" ".join(self.clusters[c]["aliases"]) for c in cluster_names])
            unk_emb = self._embed(unknown)
            # cosine similarity
            cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
            unk_n = unk_emb / (np.linalg.norm(unk_emb, axis=1, keepdims=True) + 1e-9)
            sims = np.matmul(unk_n, cent_n.T)  # [U, C]
            for i, sk in enumerate(unknown):
                ci = int(np.argmax(sims[i]))
                cname = cluster_names[ci]
                normalized[cname].append(sk)
                mapping[sk] = cname
            unknown = []

        # Dedup and sort
        for k in list(normalized.keys()):
            vals = sorted(list(dict.fromkeys(normalized[k])))
            if vals:
                normalized[k] = vals
            else:
                del normalized[k]

        return {"normalized": normalized, "mapping": mapping, "unknown": unknown}