"""app/main.py - FastAPI entry point. Run from the repo root: python -m uvicorn app.main:app --reload"""
import io
import re
from pathlib import Path

import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .feedback import SECTIONS, _jd_terms, _years_covered, generate_recommendations

app = FastAPI(title="Resume Analyzer")
INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

SKILLS = """python,java,javascript,typescript,c++,c#,golang,rust,sql,nosql,html,css,react,angular,vue,node.js,express,
django,flask,fastapi,spring boot,docker,kubernetes,aws,azure,gcp,terraform,jenkins,ci/cd,git,linux,mongodb,postgresql,
mysql,redis,kafka,spark,hadoop,pandas,numpy,scikit-learn,tensorflow,pytorch,machine learning,deep learning,nlp,
computer vision,data analysis,data visualization,statistics,tableau,power bi,excel,rest api,graphql,microservices,agile,
scrum,jira,figma,ui/ux,selenium,pytest,etl,airflow,llm,a/b testing,project management,communication,leadership,html5,bootstrap,tailwind,next.js,redux,jquery,php,laravel,ruby,rails,kotlin,swift,android,ios,flutter,oracle,dynamodb,elasticsearch,snowflake,bigquery,databricks,looker,sap,salesforce,seo,google analytics,photoshop,cybersecurity,networking,unit testing,oop,data structures,algorithms,problem solving,time series,regression,classification,opencv,hugging face,langchain,generative ai,prompt engineering,github,gitlab,postman,rabbitmq,nginx,ansible,prometheus,grafana,bash,matlab,scala,devops,cloud computing,business analysis,stakeholder management,teamwork,data structures""".replace("\n", "").split(",")


def find_skills(text):
    t = text.lower()
    return [s for s in SKILLS if re.search(rf"(?<![\w+#]){re.escape(s)}(?![\w+#])", t)]


def has_term(text, term):
    return bool(re.search(rf"(?<![\w+#]){re.escape(term)}(?![\w+#])", text.lower()))


def jd_skill_list(jd):
    """Known skills found in the JD; if few, top-up with the JD's own most important keywords."""
    skills = find_skills(jd)
    if len(skills) < 8:
        ranked, _ = _jd_terms(jd, top=40)
        for t in ranked:
            if len(skills) >= 12:
                break
            if len(t) > 3 and not any(t in x or x in t for x in skills):
                skills.append(t)
    return skills


def extract_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


@app.get("/")
def home():
    return FileResponse(INDEX)


@app.post("/analyze")
async def analyze(resume: UploadFile = File(...), job_description: str = Form(...)):
    data = await resume.read()
    if not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF resume.")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "PDF is larger than 5 MB.")
    if len(job_description.strip()) < 30:
        raise HTTPException(400, "Please paste a fuller job description.")
    try:
        text = extract_text(data)
    except Exception:
        raise HTTPException(400, "Couldn't read that PDF. Is it corrupted or password-protected?")
    if len(text.strip()) < 100:
        raise HTTPException(422, "No readable text found. Scanned/image PDFs need OCR, which isn't supported yet.")

    rl = text.lower()
    jd_skills = jd_skill_list(job_description)
    matched = [s for s in jd_skills if has_term(text, s)]
    missing = [s for s in jd_skills if s not in matched]
    extra = [s for s in find_skills(text) if s not in jd_skills]

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform([text, job_description])
    sim = float(cosine_similarity(m[0], m[1])[0][0])

    jd_years = max([int(x) for x in re.findall(r"(\d+)\+?\s*(?:years|yrs)", job_description.lower())], default=0)
    have = _years_covered(text)
    lines = [l for l in text.splitlines() if 30 <= len(l.strip()) <= 260]
    quant = sum(bool(re.search(r"\d|%|\$|₹", l)) for l in lines)

    parts = {
        "Skills match": (len(matched) / len(jd_skills) if jd_skills else min(sim / 0.4, 1), 35),
        "Keyword relevance": (min(sim / 0.4, 1), 25),
        "Experience": ((min(have / jd_years, 1) if have else 0.3) if jd_years
                       else (0.7 if re.search(SECTIONS["experience"], rl) else 0.3), 15),
        "Quantified impact": (min(quant / len(lines) / 0.4, 1) if len(lines) >= 4 else 0.5, 15),
        "Resume structure": (sum(bool(re.search(p, rl)) for p in SECTIONS.values()) / len(SECTIONS), 10),
    }
    breakdown = [{"name": k, "score": round(f * mx, 1), "max": mx} for k, (f, mx) in parts.items()]
    score = round(sum(b["score"] for b in breakdown))
    verdict = ("Excellent match" if score >= 80 else "Strong match" if score >= 65
               else "Moderate match" if score >= 45 else "Needs work")

    recs = generate_recommendations(text, job_description, matched_skills=matched or None,
                                    missing_skills=missing or None, similarity=sim)
    return {"score": score, "verdict": verdict, "breakdown": breakdown, "matched": matched,
            "missing": missing, "extra": extra[:12], "similarity": round(sim * 100),
            "words": len(text.split()), "recommendations": recs}