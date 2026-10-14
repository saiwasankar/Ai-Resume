"""Input-quality checks that prevent generic text from receiving a resume score."""
import re

JD_CONTEXT = re.compile(
    r"\b(job title|job description|responsibilities|qualifications|requirements|"
    r"position|candidate|hiring|we are looking for|we're looking for|you will|"
    r"ideal candidate|software engineer|data engineer|data analyst|developer|"
    r"designer|product manager|project manager|scientist|specialist|internship|intern)\b",
    re.I,
)
JD_WORK = re.compile(
    r"\b(develop|design|build|implement|maintain|manage|lead|analy[sz]e|"
    r"support|deliver|create|deploy|improve|collaborate|own|responsible|"
    r"experience|proficient|required|must have|work with|work on)\w*\b",
    re.I,
)

SECTION_PATTERNS = {
    "summary": r"(?:professional\s+)?summary|profile|objective",
    "experience": r"(?:work|professional)\s+experience|experience|employment|internships?",
    "education": r"education|academic background",
    "skills": r"(?:technical|core)\s+skills|skills|competencies",
    "projects": r"(?:academic|personal)\s+projects|projects",
    "certifications": r"certifications?|licenses",
}


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z+#.\-/]{1,}", text.lower())


def is_job_description(text: str, known_skill_count: int) -> bool:
    tokens = word_tokens(text)
    if len(tokens) < 10 or len(set(tokens)) < 8:
        return False
    # Require both a hiring/role signal and language describing work or qualifications.
    # Several known skills can serve as context for short, compact job ads.
    return bool(JD_CONTEXT.search(text) and (JD_WORK.search(text) or known_skill_count >= 3))


def resume_section_headings(text: str) -> set[str]:
    found = set()
    for line in text.splitlines():
        heading = re.sub(r"^[\s•|·▪◦\-–—]+|[\s:|·▪◦\-–—]+$", "", line).strip().lower()
        if not heading or len(heading) > 70:
            continue
        for section, pattern in SECTION_PATTERNS.items():
            if re.fullmatch(pattern, heading):
                found.add(section)
    return found


def is_resume(text: str) -> bool:
    tokens = word_tokens(text)
    if len(tokens) < 80 or len(set(tokens)) < 35:
        return False
    headings = resume_section_headings(text)
    has_contact = bool(
        re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text)
        or re.search(r"(?:\+?\d[\d .()\-]{8,}\d)", text)
    )
    has_dates = bool(re.search(r"\b(?:19|20)\d{2}\s*(?:(?:-|–|to)\s*(?:(?:19|20)\d{2}|present|current))\b", text, re.I))
    return len(headings) >= 2 or (len(headings) >= 1 and (has_contact or has_dates))
