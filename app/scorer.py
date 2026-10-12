import re
from collections import Counter

STOPWORDS = {
    "the", "and", "for", "with", "you", "our", "are", "will", "this", "that",
    "have", "has", "your", "from", "who", "all", "can", "about", "their",
    "they", "any", "such", "also", "more", "work", "team", "role", "job",
    "experience", "years", "ability", "strong", "good", "must", "including",
    "etc", "new", "use", "using", "well", "within", "across", "other",
}

ACTION_VERBS = [
    "developed", "built", "designed", "led", "managed", "implemented",
    "created", "optimized", "improved", "launched", "deployed",
    "automated", "analyzed", "delivered", "reduced", "increased",
]

SECTIONS = {
    "summary": ["summary", "objective", "profile"],
    "experience": ["experience", "employment", "internship"],
    "education": ["education"],
    "skills": ["skills"],
    "projects": ["projects"],
}


def _jd_keywords(jd_text: str, top: int = 25) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", jd_text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(words).most_common(top)]


def score_skills(skills_result: dict) -> float:
    return (skills_result.get("match_percentage", 0) / 100) * 30


def score_keywords(resume_text: str, jd_text: str) -> float:
    keywords = _jd_keywords(jd_text)

    if not keywords:
        return 0.0

    resume = resume_text.lower()
    hits = sum(1 for keyword in keywords if keyword in resume)

    return (hits / len(keywords)) * 15


def score_experience(resume_text: str) -> float:
    text = resume_text.lower()
    points = 0

    if any(key in text for key in SECTIONS["experience"]):
        points += 5

    points += min(sum(1 for verb in ACTION_VERBS if verb in text), 5)

    if re.search(r"\b(?:19|20)\d{2}\b", text):
        points += 5

    return min(points, 15)


def score_quantified(resume_text: str) -> float:
    pattern = (
        r"\d+(?:\.\d+)?\s?%"
        r"|\$\s?\d+(?:,\d{3})*(?:\.\d+)?"
        r"|\b\d+\s?(?:users|clients|customers|projects|members|requests)\b"
        r"|\b\d{2,}\+"
    )

    hits = len(re.findall(pattern, resume_text.lower()))
    return min(hits / 5, 1) * 7.5


def score_structure(resume_text: str) -> float:
    text = resume_text.lower()

    found = sum(
        1
        for keys in SECTIONS.values()
        if any(key in text for key in keys)
    )

    points = (found / len(SECTIONS)) * 4.5
    word_count = len(resume_text.split())

    if 300 <= word_count <= 900:
        points += 3
    elif 150 <= word_count < 300 or 900 < word_count <= 1200:
        points += 1.5

    return min(points, 7.5)


def calculate_score(
    resume_text: str,
    jd_text: str,
    skills_result: dict,
    semantic_pct: float = 0.0,
) -> dict:
    # Accept a percentage between 0 and 100.
    similarity = max(0.0, min(float(semantic_pct), 100.0))
    similarity_points = (similarity / 100) * 25

    parts = [
        ("Skills match", score_skills(skills_result), 30),
        ("Keyword relevance", score_keywords(resume_text, jd_text), 15),
        ("Text similarity", similarity_points, 25),
        ("Experience", score_experience(resume_text), 15),
        ("Quantified achievements", score_quantified(resume_text), 7.5),
        ("Resume structure", score_structure(resume_text), 7.5),
    ]

    breakdown = [
        {"label": name, "score": round(score, 1), "max": maximum}
        for name, score, maximum in parts
    ]

    total = round(sum(item["score"] for item in breakdown))

    return {
        "score": min(total, 100),
        "breakdown": breakdown,
    }