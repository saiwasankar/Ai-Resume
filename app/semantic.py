from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(resume_text: str, job_text: str) -> float:
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_text])

    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)


def compare_text(resume_text: str, job_text: str) -> dict:
    score = tfidf_similarity(resume_text, job_text)

    return {
        "tfidf_similarity": score,
        "semantic_similarity": score,
    }