"""Semantic comparison with a small, general-purpose pretrained sentence model.

The model is not trained on resumes and this app does not collect or learn from
uploaded files. When sentence-transformers is unavailable, comparison falls back
to TF-IDF and reports that method accurately to the caller.
"""
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def compare_text(resume_text: str, job_text: str) -> dict:
    if not resume_text.strip() or not job_text.strip():
        return {"similarity": 0.0, "analysis_method": "No readable text"}

    try:
        model = _embedding_model()
        vectors = model.encode([resume_text, job_text], normalize_embeddings=True)
        score = float(vectors[0] @ vectors[1])
        return {
            "similarity": round(max(0.0, min(score, 1.0)), 4),
            "analysis_method": "Pretrained semantic embeddings + evidence checks",
        }
    except Exception:
        # Preserve a usable local path in offline/minimal installations.
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            vectors = vectorizer.fit_transform([resume_text, job_text])
            score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except ValueError:  # Empty vocabulary (for example punctuation-only input).
            score = 0.0
        return {
            "similarity": round(score, 4),
            "analysis_method": "TF-IDF fallback + evidence checks",
        }


def compare_to_many(reference: str, candidates: list[str]) -> list[float]:
    """Compare one criterion with many resume lines in one model call."""
    if not reference.strip() or not candidates:
        return [0.0] * len(candidates)
    try:
        model = _embedding_model()
        vectors = model.encode([reference, *candidates], normalize_embeddings=True)
        return [round(max(0.0, min(float(vectors[0] @ vector), 1.0)), 4) for vector in vectors[1:]]
    except Exception:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            vectors = vectorizer.fit_transform([reference, *candidates])
            scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
            return [round(float(score), 4) for score in scores]
        except ValueError:
            return [0.0] * len(candidates)
