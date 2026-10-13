"""
feedback_engine.py - recommendations built from the actual resume + job description.

No templates-per-score and no training data needed: every recommendation is derived from
evidence (JD terms, JD sentences, resume lines, section/number/years analysis), so different
resume/JD pairs produce different advice. Same inputs -> same output (deterministic).
"""
import re
from collections import Counter
from datetime import date

STOP = set("""a an the and or of to in for with on at by from as is are be this that will you your we our their it its
experience work working team teams ability strong skills skill years year etc including preferred required plus role job
candidate must have has can able good great new using use based across within into such other more also any all not
looking join company position responsibilities requirements qualifications""".split())

WEAK = ("responsible for", "worked on", "helped", "assisted", "involved in", "participated in",
        "duties included", "tasked with")

SECTIONS = {
    "summary": r"\b(summary|objective|profile)\b",
    "experience": r"\b(experience|employment|internship)\b",
    "projects": r"\bprojects?\b",
    "education": r"\beducation\b",
    "skills": r"\bskills\b",
    "certifications": r"\b(certifications?|certificates?)\b",
}


# ---------- helpers ----------
def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 15]


def _tokens(text):
    return [t.strip(".-") for t in re.findall(r"[a-zA-Z][a-zA-Z+#.\-]+", text.lower())]


def _cut(s, n=140):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _jd_terms(jd, top=40):
    toks = [t for t in _tokens(jd) if t and t not in STOP and len(t) > 2]
    uni = Counter(toks)
    bi = Counter(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    terms = uni + Counter({k: v for k, v in bi.items() if v >= 2})
    return [t for t, _ in terms.most_common(top)], terms


def _term_info(term, jd_sents):
    """(sentence_count, is_required, first_sentence_containing_term)"""
    hits = [s for s in jd_sents if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", s.lower())]
    required = any(re.search(r"\b(required|must|essential|mandatory|minimum|proficien\w+)\b", s.lower()) for s in hits)
    return len(hits), required, (hits[0] if hits else "")


def _years_covered(text):
    now = date.today().year
    covered = set()
    for a, b in re.findall(r"((?:19|20)\d\d)\s*(?:-|–|to)\s*((?:19|20)\d\d|present|current)", text.lower()):
        start = int(a)
        end = now if b in ("present", "current") else int(b)
        covered.update(range(start, max(start, end)))
    explicit = [int(x) for x in re.findall(r"(\d+)\+?\s*(?:years|yrs)", text.lower())]
    return max(len(covered), max(explicit, default=0))


# ---------- main ----------
def generate_recommendations(resume_text, jd_text, matched_skills=None, missing_skills=None,
                             similarity=None, max_items=7):
    """
    resume_text, jd_text : raw text
    matched_skills / missing_skills : optional lists from your existing skills.py (used if given)
    similarity : optional 0-1 TF-IDF score from your semantic.py
    Returns list of dicts: {priority, category, title, detail, evidence}
    """
    resume_l = resume_text.lower()
    jd_sents = _sentences(jd_text)
    ranked, counts = _jd_terms(jd_text)

    def present(t):
        return bool(re.search(rf"(?<!\w){re.escape(t.lower())}(?!\w)", resume_l))

    def weight(t):
        n, req, _ = _term_info(t.lower(), jd_sents)
        return counts.get(t.lower(), 1) + n + (3 if req else 0)

    if missing_skills is None:
        missing_skills = [t for t in ranked if not present(t)]
    if matched_skills is None:
        matched_skills = [t for t in ranked if present(t)]
    missing = sorted(set(missing_skills), key=weight, reverse=True)
    matched = sorted(set(matched_skills), key=weight, reverse=True)

    recs = []

    def add(impact, category, title, detail, evidence=""):
        recs.append({"impact": impact, "category": category, "title": title,
                     "detail": detail, "evidence": evidence})

    # 1. Missing skills, each tied to what the JD actually says about it
    for t in missing[:4]:
        n, req, snippet = _term_info(t.lower(), jd_sents)
        stress = "listed as a requirement" if req else f"mentioned in {n} place(s)" if n else "relevant to the role"
        add(60 + weight(t) * 2 + (15 if req else 0), "Missing skill", f"Add evidence of “{t}”",
            f"The job description has “{t}” {stress}, but your resume never mentions it. If you've used it, "
            f"put it inside a bullet showing what you built or improved with it. If not, a small project or "
            f"certification using it is the fastest way to close the gap.",
            _cut(snippet) if snippet else "")

    # 2. Experience length vs JD ask
    jd_years = max([int(x) for x in re.findall(r"(\d+)\+?\s*(?:years|yrs)", jd_text.lower())], default=0)
    have_years = _years_covered(resume_text)
    if jd_years and have_years and have_years < jd_years:
        add(75, "Experience", f"Bridge the experience gap ({have_years} vs {jd_years}+ years)",
            f"The JD asks for {jd_years}+ years; your dates add up to roughly {have_years}. Don't pad - instead "
            f"surface scope and ownership (team size, systems, budgets) and include relevant projects, "
            f"internships or freelance work so total relevant experience is visible.")
    elif jd_years and not have_years:
        add(55, "Experience", "Make your experience duration explicit",
            f"The JD asks for {jd_years}+ years but I couldn't find clear date ranges. Add start-end dates "
            f"to each role so screeners can verify the match quickly.")

    # 3. Weak bullets, quoting the candidate's own lines
    lines = [l.strip(" •-*\t") for l in resume_text.splitlines() if 30 <= len(l.strip()) <= 260]
    weak_lines = [l for l in lines if any(w in l.lower() for w in WEAK)]
    if weak_lines:
        target = missing[0] if missing else (matched[0] if matched else "a key tool from the JD")
        add(50 + 5 * min(len(weak_lines), 4), "Impact wording",
            f"Rewrite {len(weak_lines)} passive bullet(s)",
            f"These lines describe duties, not results. Start with a strong verb and end with an outcome - "
            f"e.g. “Built … using {target}, cutting X by Y%”.",
            _cut(weak_lines[0]))

    # 4. Quantification, with a real unquantified line as the example
    if len(lines) >= 4:
        quant = [l for l in lines if re.search(r"\d|%|\$|₹", l)]
        ratio = len(quant) / len(lines)
        if ratio < 0.3:
            plain = next((l for l in lines if l not in quant and not any(w in l.lower() for w in WEAK)), "")
            add(55 + int((0.3 - ratio) * 60), "Quantification",
                f"Only {len(quant)} of {len(lines)} bullets contain numbers",
                "Add scale, speed, accuracy, users, cost or time saved to more bullets. If you lack exact "
                "figures, use honest estimates (“~30 users”, “reduced manual work from hours to minutes”).",
                _cut(plain))

    # 5. Missing sections, prioritised by what this JD emphasises
    jd_l = jd_text.lower()
    for name, pat in SECTIONS.items():
        if not re.search(pat, resume_l):
            wanted = {"certifications": "certif", "projects": "project", "skills": "skill"}.get(name)
            boost = 15 if wanted and wanted in jd_l else 0
            if name in ("experience", "education", "skills") or boost:
                add(40 + boost, "Structure", f"No clear “{name.title()}” section found",
                    f"ATS parsers look for a labelled {name} section"
                    + (f", and this JD explicitly refers to {wanted}…" if boost else ".")
                    + " Add a plainly-named heading.")

    # 6. Strengths to lead with
    if matched:
        top = ", ".join(matched[:4])
        add(35, "Strengths", f"Lead with your strongest overlap: {top}",
            "These are the JD terms you already cover that the JD stresses most. Move them into the top third "
            "of page one (summary + first role bullets) so they're seen first.")

    # 7. Low similarity -> use the JD's own vocabulary
    absent_terms = [t for t in ranked if not present(t)][:5]
    if (similarity is not None and similarity < 0.25 and absent_terms) or len(absent_terms) >= 4:
        add(45 + (int((0.25 - similarity) * 100) if similarity is not None and similarity < 0.25 else 0),
            "Vocabulary", "Mirror the JD's wording where it's truthful",
            f"Frequently used JD terms missing from your resume: {', '.join(absent_terms)}. Use the same "
            f"terminology for work you've genuinely done (e.g. say what the JD says, not a synonym).")

    recs.sort(key=lambda r: r["impact"], reverse=True)
    out = []
    for r in recs[:max_items]:
        r["priority"] = "High" if r["impact"] >= 75 else "Medium" if r["impact"] >= 50 else "Low"
        del r["impact"]
        out.append(r)
    return out