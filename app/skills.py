import re

SKILL_DB = [
    "python", "java", "javascript", "typescript", "react", "node.js",
    "sql", "mongodb", "postgresql", "aws", "docker", "kubernetes",
    "git", "machine learning", "deep learning", "nlp", "pandas",
    "numpy", "tensorflow", "pytorch", "fastapi", "django", "flask",
    "rest api", "ci/cd", "agile", "scrum", "excel", "power bi",
    "tableau", "communication", "leadership", "problem solving",
]

def extract_skills(text: str) -> set[str]:
    text = text.lower()
    found = set()
    for skill in SKILL_DB:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(skill)
    return found

def compare_skills(resume_text: str, jd_text: str) -> dict:
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    match_pct = (len(matched) / len(jd_skills) * 100) if jd_skills else 0
    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "match_percentage": round(match_pct, 1),
    }