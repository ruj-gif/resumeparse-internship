from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class MatchResult:
    index: int
    score: float
    keywords: List[str]


class InternshipMatcher:
    """
    TF-IDF + cosine similarity matcher.
    Fit on internship texts once, then match many resumes.
    """

    def __init__(
        self,
        *,
        stop_words: str | None = "english",
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: Optional[int] = 40000,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
        )
        self._internship_matrix = None
        self._internship_texts: List[str] = []

    def fit(self, internship_texts: Sequence[str]) -> "InternshipMatcher":
        texts = [t if isinstance(t, str) else "" for t in internship_texts]
        self._internship_texts = texts
        self._internship_matrix = self.vectorizer.fit_transform(texts)
        return self

    def _top_shared_keywords(
        self, resume_vec, internship_vec, *, top_n: int = 8
    ) -> List[str]:
        """
        Explain match using top features with highest (resume_tfidf * internship_tfidf).
        Works well for highlighting overlapping terms/phrases in TF-IDF space.
        """
        if resume_vec.nnz == 0 or internship_vec.nnz == 0:
            return []
        shared = resume_vec.multiply(internship_vec)
        if shared.nnz == 0:
            return []

        coo = shared.tocoo()
        # pick indices of top weights
        order = np.argsort(coo.data)[::-1][:top_n]
        feat_idx = coo.col[order]
        weights = coo.data[order]
        names = self.vectorizer.get_feature_names_out()
        out: List[str] = []
        for i, w in zip(feat_idx, weights):
            term = str(names[i]).strip()
            if term and term not in out:
                out.append(term)
        return out

    def match_top_k(self, resume_text: str, *, top_k: int = 3) -> List[MatchResult]:
        if self._internship_matrix is None:
            raise RuntimeError("Call fit() before match_top_k().")
        resume_text = resume_text if isinstance(resume_text, str) else ""
        resume_vec = self.vectorizer.transform([resume_text])
        sims = cosine_similarity(resume_vec, self._internship_matrix).ravel()
        if sims.size == 0:
            return []

        top_idx = np.argsort(sims)[::-1][:top_k]
        results: List[MatchResult] = []
        for idx in top_idx:
            internship_vec = self._internship_matrix[idx]
            keywords = self._top_shared_keywords(resume_vec, internship_vec, top_n=10)
            results.append(MatchResult(index=int(idx), score=float(sims[idx]), keywords=keywords))
        return results


def greedy_allocate_with_capacity(
    scores: np.ndarray,
    capacity: Optional[Sequence[int]] = None,
) -> List[Optional[int]]:
    """
    Allocate internships to resumes (one per resume) with optional capacities.
    Greedy by highest score across all pairs.

    scores: shape (n_resumes, n_internships)
    returns: list of allocated internship index per resume (or None)
    """
    n_resumes, n_internships = scores.shape
    cap = np.array(list(capacity), dtype=int) if capacity is not None else np.ones(n_internships, dtype=int)
    cap = np.maximum(cap, 0)

    allocated: List[Optional[int]] = [None] * n_resumes
    used_pairs: List[Tuple[int, int, float]] = []

    for r in range(n_resumes):
        for i in range(n_internships):
            used_pairs.append((r, i, float(scores[r, i])))
    used_pairs.sort(key=lambda x: x[2], reverse=True)

    remaining_resumes = set(range(n_resumes))
    for r, i, _s in used_pairs:
        if r not in remaining_resumes:
            continue
        if cap[i] <= 0:
            continue
        allocated[r] = i
        cap[i] -= 1
        remaining_resumes.remove(r)
        if not remaining_resumes:
            break

    return allocated

