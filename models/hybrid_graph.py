"""
Hybrid Graph–Semantic Matcher for ResuMatch-X
- Builds a bipartite graph between one resume node and internship/job nodes
- Edge weights combine lexical TF-IDF, semantic MiniLM/SBERT, and skill overlap
- Ranks jobs via personalized PageRank and blends with base similarities
- Provides explainability payloads
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import hashlib
import math

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _hash_text(txt: str) -> str:
    h = hashlib.sha256((txt or "").encode("utf-8")).hexdigest()
    return h[:16]


class HybridGraphMatcher:
    """
    Construct a hybrid graph using precomputed TF-IDF and SBERT embeddings from EnhancedResuMatchModel.
    """

    def __init__(self, base_model):
        self.base = base_model  # EnhancedResuMatchModel
        if not getattr(self.base, "is_fitted", False):
            raise ValueError("Base model must be fitted before using HybridGraphMatcher")

    def _resume_vectors(self, resume_text: str) -> Tuple[np.ndarray, np.ndarray]:
        # TF-IDF vector
        tfv = self.base.tfidf_vectorizer.transform([resume_text])
        # SBERT vector (optional)
        sbert = None
        if self.base.sentence_transformer is not None and self.base.job_embeddings_sbert is not None:
            sbert = self.base.sentence_transformer.encode([resume_text])
        return tfv, sbert

    def build_graph(self, resume_text: str, resume_skills: List[str]) -> Tuple[nx.Graph, Dict[str, Any]]:
        G = nx.Graph()
        resume_id = f"resume::{_hash_text(resume_text)}"
        G.add_node(resume_id, type="resume")

        tfv, sbert = self._resume_vectors(resume_text)
        job_texts = [j['combined_text'] for j in self.base.job_descriptions]

        # Lexical similarities (dense array)
        tfidf_sim = cosine_similarity(tfv, self.base.job_embeddings_tfidf).flatten()
        # Semantic similarities (optional)
        if sbert is not None and self.base.job_embeddings_sbert is not None:
            sbert_sim = cosine_similarity(sbert, self.base.job_embeddings_sbert).flatten()
        else:
            sbert_sim = np.zeros(len(self.base.job_descriptions), dtype=float)

        # Skill overlap score (Jaccard)
        res_sk = {s.strip().lower() for s in (resume_skills or []) if isinstance(s, str)}

        explain = {"edges": []}
        for idx, job in enumerate(self.base.job_descriptions):
            jid = f"job::{job['id']}"
            G.add_node(jid, type="job", title=job['title'], company=job['company'])

            job_sk = {s.strip().lower() for s in (job.get('skills_needed') or []) if isinstance(s, str)}
            if not job_sk:
                # Light extraction fallback from description/requirements already done in base model at predict time,
                # but here we recompute a simple proxy.
                job_sk = set()
            inter = len(res_sk & job_sk)
            union = len(res_sk | job_sk) if (res_sk or job_sk) else 0
            skill_jaccard = (inter / union) if union else 0.0

            # Edge weight combines normalized components
            w_lex = float(tfidf_sim[idx])
            w_sem = float(sbert_sim[idx])
            w_skill = float(skill_jaccard)
            # Normalize to [0,1]
            w_lex_n = max(0.0, min(1.0, w_lex))
            w_sem_n = max(0.0, min(1.0, w_sem))
            w_skill_n = max(0.0, min(1.0, w_skill))

            # Store per-edge features; actual blending is deferred to rank()
            G.add_edge(resume_id, jid, tfidf=w_lex_n, sbert=w_sem_n, skill=w_skill_n)
            explain["edges"].append({
                "job_id": job['id'],
                "tfidf": w_lex_n,
                "sbert": w_sem_n,
                "skill": w_skill_n,
            })

        return G, {"resume_id": resume_id, **explain}

    def rank(self,
             G: nx.Graph,
             resume_id: str,
             weights: Dict[str, float],
             top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Run personalized PageRank with composite edge weights, then blend with base similarities.
        weights: {lexical, semantic, skill, graph}
        """
        w_lex = float(weights.get("lexical", 0.4))
        w_sem = float(weights.get("semantic", 0.4))
        w_skill = float(weights.get("skill", 0.2))
        w_graph = float(weights.get("graph", 0.3))

        # Compose edge weight attribute used by PageRank
        for u, v, d in G.edges(data=True):
            d["weight"] = max(1e-6, w_lex * d.get("tfidf", 0.0) + w_sem * d.get("sbert", 0.0) + w_skill * d.get("skill", 0.0))

        # Personalized PageRank starting at the resume node
        pr = nx.pagerank(G, alpha=0.85, personalization={resume_id: 1.0}, weight="weight")

        # Collect job nodes and scores
        scored: List[Tuple[str, float]] = []
        for n, s in pr.items():
            if n.startswith("job::"):
                scored.append((n, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for node, gscore in scored[: max(top_k*2, top_k)]:
            job_id = node.split("::", 1)[1]
            job = next((j for j in self.base.job_descriptions if str(j['id']) == str(job_id)), None)
            if not job:
                continue
            # Retrieve original sims for transparent breakdown
            # Recompute local edge features (graph kept them on edge)
            edge = G.get_edge_data(resume_id, node)
            tfidf = float(edge.get("tfidf", 0.0)) if edge else 0.0
            sbert = float(edge.get("sbert", 0.0)) if edge else 0.0
            skill = float(edge.get("skill", 0.0)) if edge else 0.0

            base_blend = w_lex * tfidf + w_sem * sbert + w_skill * skill
            final_score = (1 - w_graph) * base_blend + w_graph * float(gscore)

            results.append({
                "job_id": job['id'],
                "title": job['title'],
                "company": job['company'],
                "location": job.get('location', ''),
                "similarity_score": float(final_score),
                "graph_score": float(gscore),
                "breakdown": {
                    "tfidf": tfidf,
                    "sbert": sbert,
                    "skill": skill,
                    "weights": {"lexical": w_lex, "semantic": w_sem, "skill": w_skill, "graph": w_graph}
                }
            })

            if len(results) >= top_k:
                break

        return results