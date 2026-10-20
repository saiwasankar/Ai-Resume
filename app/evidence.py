"""Extract job criteria and link them to resume passages with conservative evidence."""
import re

from .semantic import compare_to_many
from .reranker import trained_score

GENERIC = {
    "experience", "years", "ability", "strong", "good", "excellent", "skills",
    "knowledge", "preferred", "required", "requirements", "responsibilities",
    "qualification", "qualifications", "candidate", "team", "work", "role",
    "including", "such", "using", "with", "from", "that", "this", "will",
}
ACTION = re.compile(
    r"\b(develop|design|build|implement|maintain|manage|lead|analy[sz]e|support|"
    r"deliver|create|deploy|improve|collaborate|own|responsible|proficient|"
    r"experience|familiar|knowledge|using|work with|work on|ability to)\w*\b", re.I
)
SKILL_WORDS = re.compile(
    r"\b(python|java|javascript|typescript|c\+\+|c#|golang|rust|sql|react|angular|"
    r"vue|node\.js|django|flask|fastapi|docker|kubernetes|aws|azure|gcp|terraform|"
    r"mongodb|postgresql|mysql|redis|kafka|spark|pandas|numpy|scikit-learn|tensorflow|"
    r"pytorch|machine learning|deep learning|nlp|computer vision|tableau|power bi|"
    r"excel|rest api|graphql|microservices|agile|scrum|jira|selenium|pytest|etl|"
    r"airflow|llm|a/b testing|project management|communication|leadership|"
    r"data analysis|statistics|linux|git|ci/cd|generative ai|prompt engineering)\b", re.I
)


def _clean(line: str) -> str:
    return re.sub(r"^[\s•●▪◦*\-–—\d.)]+|\s+", " ", line).strip(" :;•-–—")


def extract_requirements(job_text: str, recognized_skills: list[str], limit: int = 24) -> list[dict]:
    """Return deduplicated job criteria from explicit bullet/sentence statements."""
    candidates = []
    for raw in re.split(r"[\n\r]+|(?<=[.!?])\s+", job_text):
        text = _clean(raw)
        if len(text.split()) < 3 or len(text) > 360:
            continue
        if ACTION.search(text) or SKILL_WORDS.search(text):
            candidates.append(text)
    # Keep explicit skill requirements even when the JD uses compact comma lists.
    for skill in recognized_skills:
        if not any(re.search(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])", c, re.I) for c in candidates):
            candidates.append(f"Experience with {skill}")
    seen, result = set(), []
    for text in candidates:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        terms = [w for w in re.findall(r"[a-z][a-z+#.\-/]{1,}", text.lower()) if w not in GENERIC]
        if len(terms) < 2 and not SKILL_WORDS.search(text):
            continue
        kind = "Skill" if SKILL_WORDS.search(text) and len(text.split()) <= 12 else "Experience / responsibility"
        result.append({"text": text, "type": kind, "terms": terms})
        if len(result) >= limit:
            break
    return result


def analyze_requirements(job_text: str, resume_text: str, recognized_skills: list[str]) -> list[dict]:
    requirements = extract_requirements(job_text, recognized_skills)
    lines = [re.sub(r"\s+", " ", line).strip() for line in resume_text.splitlines()]
    lines = [line for line in lines if 25 <= len(line) <= 360]
    if not lines:
        lines = [re.sub(r"\s+", " ", resume_text[i:i + 300]).strip() for i in range(0, len(resume_text), 240)]
    lines = lines[:120]
    results = []
    for req in requirements:
        requirement = req["text"]
        req_terms = set(req["terms"])
        similarities = compare_to_many(requirement, lines)
        best = {"line": "", "similarity": 0.0, "overlap": 0}
        for line, sim in zip(lines, similarities):
            words = set(re.findall(r"[a-z][a-z+#.\-/]{1,}", line.lower())) - GENERIC
            overlap = len(req_terms & words)
            # Similarity alone can be misleading; require concrete lexical evidence too.
            if (overlap, sim) > (best["overlap"], best["similarity"]):
                best = {"line": line, "similarity": sim, "overlap": overlap}
        explicit_skill = next((s for s in recognized_skills if re.search(rf"(?<![\w+#]){re.escape(s)}(?![\w+#])", requirement, re.I)), None)
        exact_skill = bool(explicit_skill and re.search(rf"(?<![\w+#]){re.escape(explicit_skill)}(?![\w+#])", resume_text, re.I))
        model_probability = trained_score(requirement, best["line"]) if best["line"] else None
        semantic_supported = (best["overlap"] >= 2 and best["similarity"] >= 0.30) or (best["overlap"] >= 3 and best["similarity"] >= 0.18)
        # A trained model can raise or lower confidence, but it cannot override
        # the requirement that a concrete resume passage exists.
        supported = exact_skill or (bool(best["line"]) and ((model_probability is not None and model_probability >= 0.70) or semantic_supported))
        high_confidence = exact_skill or (model_probability is not None and model_probability >= 0.85) or (best["overlap"] >= 3 and best["similarity"] >= 0.3)
        results.append({
            "requirement": requirement,
            "type": req["type"],
            "status": "supported" if supported else "not evidenced",
            "confidence": "high" if high_confidence else "moderate" if supported else "low",
            "evidence": best["line"][:300] if supported else "",
            "evidence_similarity": round(best["similarity"], 3),
            "model_probability": model_probability,
        })
    return results
