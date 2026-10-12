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
    return skills_result["match_percentage"] / 100 * 40


def score_keywords(resume_text: str, jd_text: str) -> float:
    keywords = _jd_keywords(jd_text)
    if not keywords:
        return 0
    resume = resume_text.lower()
    hits = sum(1 for k in keywords if k in resume)
    return hits / len(keywords) * 20


def score_experience(resume_text: str) -> float:
    t = resume_text.lower()
    pts = 0
    if any(k in t for k in SECTIONS["experience"]):
        pts += 8
    pts += min(sum(1 for v in ACTION_VERBS if v in t), 6)
    if re.search(r"(19|20)\d{2}", t):
        pts += 6
    return min(pts, 20)


def score_quantified(resume_text: str) -> float:
    pattern = r"\d+\s?%|\$\s?\d+|\d+\s?(users|clients|customers|projects|members|requests)|\b\d{2,}\+"
    hits = len(re.findall(pattern, resume_text.lower()))
    return min(hits / 5, 1) * 10


def score_structure(resume_text: str) -> float:
    t = resume_text.lower()
    found = sum(1 for keys in SECTIONS.values() if any(k in t for k in keys))
    pts = found / len(SECTIONS) * 6
    words = len(resume_text.split())
    if 300 <= words <= 900:
        pts += 4
    elif 150 <= words < 300 or 900 < words <= 1200:
        pts += 2
    return pts


def calculate_score(resume_text: str, jd_text: str, skills_result: dict) -> dict:
    parts = [
        ("Skills match", score_skills(skills_result), 40),
        ("Keyword relevance", score_keywords(resume_text, jd_text), 20),
        ("Experience", score_experience(resume_text), 20),
        ("Quantified achievements", score_quantified(resume_text), 10),
        ("Resume structure", score_structure(resume_text), 10),
    ]
    breakdown = [{"label": n, "score": round(s, 1), "max": m} for n, s, m in parts]
    total = round(sum(b["score"] for b in breakdown))
    return {"score": total, "breakdown": breakdown}