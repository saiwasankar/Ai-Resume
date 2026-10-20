"""RoleFit API and web entry point. Interactive OpenAPI docs are served at /docs."""
import io
import json
import re
from pathlib import Path
from contextlib import asynccontextmanager

import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .database import connection, init_db
from .feedback import SECTIONS, _years_covered, generate_recommendations
from .evidence import analyze_requirements
from .schemas import JobCreate, JobUpdate
from .semantic import compare_text
from .reranker import model_status
from .validation import is_job_description, is_resume, resume_section_headings


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(
    title="RoleFit Resume Intelligence API",
    description="Private, local-first resume diagnostics, job-fit analysis, and saved job tracking.",
    version="2.0.0",
    lifespan=lifespan,
)
INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
MAX_PDF_BYTES = 5 * 1024 * 1024
JOB_STATUSES = ("saved", "applied", "interview", "offer", "rejected", "archived")

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
    """Only return recognized skills. General JD terms are never labelled skills."""
    return find_skills(jd)


def skill_priority_report(jd: str, resume: str, skills: list[str]) -> list[dict]:
    """Rank role skills from job language, then mark resume evidence."""
    required = re.compile(r"\b(required|must|required to|mandatory|essential|core|minimum|need to)\b", re.I)
    preferred = re.compile(r"\b(preferred|nice to have|bonus|plus|desirable|ideally)\b", re.I)
    report = []
    for skill in skills:
        positions = [m.start() for m in re.finditer(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])", jd, re.I)]
        contexts = [jd[max(0, p - 120):p + len(skill) + 120] for p in positions]
        frequency = min(len(positions), 3) / 3
        score = 0.35 + frequency * 0.20
        if any(required.search(context) for context in contexts):
            score += 0.45
        elif any(preferred.search(context) for context in contexts):
            score += 0.10
        score = round(min(score, 1.0), 2)
        level = "high" if score >= 0.75 else "medium" if score >= 0.50 else "low"
        present = has_term(resume, skill)
        report.append({"skill": skill, "priority": level, "priority_score": score, "status": "matched" if present else "missing"})
    return sorted(report, key=lambda item: item["priority_score"], reverse=True)


def extract_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


@app.get("/")
def home():
    return FileResponse(INDEX)


@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "service": "RoleFit", "version": app.version, **model_status()}


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze(resume: UploadFile = File(...), job_description: str = Form(...)):
    data = await resume.read(MAX_PDF_BYTES + 1)
    if not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF resume.")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF is larger than 5 MB.")
    if len(job_description.strip()) < 30:
        raise HTTPException(400, "Please paste a fuller job description.")
    jd_skills = jd_skill_list(job_description)
    if not is_job_description(job_description, len(jd_skills)):
        raise HTTPException(
            422,
            "This text doesn't look like a job description. Paste the actual role title, responsibilities, or required qualifications before analyzing a resume.",
        )
    try:
        if not data.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF header")
        text = extract_text(data)
    except Exception:
        raise HTTPException(400, "Couldn't read that PDF. Is it corrupted or password-protected?")
    if len(text.strip()) < 100:
        raise HTTPException(422, "No readable text found. Scanned/image PDFs need OCR, which isn't supported yet.")
    headings = resume_section_headings(text)
    if not is_resume(text):
        raise HTTPException(
            422,
            "This PDF doesn't look like a structured resume. Upload a resume with clearly labeled sections such as Experience, Skills, Education, or Projects.",
        )

    rl = text.lower()
    matched = [s for s in jd_skills if has_term(text, s)]
    missing = [s for s in jd_skills if s not in matched]
    skill_priorities = skill_priority_report(job_description, text, jd_skills)
    extra = [s for s in find_skills(text) if s not in jd_skills]
    resume_lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    skill_evidence = {
        skill: next((line[:260] for line in resume_lines if has_term(line, skill)), "")
        for skill in matched
    }

    semantic = compare_text(text, job_description)
    sim = semantic["similarity"]

    jd_years = max([int(x) for x in re.findall(r"(\d+)\+?\s*(?:years|yrs)", job_description.lower())], default=0)
    have = _years_covered(text)
    lines = [l for l in text.splitlines() if 30 <= len(l.strip()) <= 260]
    quant = sum(bool(re.search(r"\d|%|\$|₹", l)) for l in lines)

    requirements = analyze_requirements(job_description, text, jd_skills)
    supported_count = sum(item["status"] == "supported" for item in requirements)
    coverage = supported_count / len(requirements) if requirements else 0.0
    # The summary score now reflects only traceable job-criterion evidence.
    # Structure and formatting remain diagnostics and do not inflate job fit.
    breakdown = [{"name": "Job requirements with resume evidence", "score": round(coverage * 100, 1), "max": 100}]
    score = round(coverage * 100)
    # A near-zero similarity and no shared role skills is a likely wrong-file or
    # wrong-role pairing. Surface it plainly and prevent a misleading high score.
    mismatch = bool(requirements) and supported_count == 0 and sim < 0.16
    if mismatch:
        raise HTTPException(
            422,
            "This resume doesn't appear relevant to the job description. Check that both documents are correct, then try again. No match score was generated.",
        )
    verdict = ("Strong evidence" if score >= 75 else "Some evidence" if score >= 40
               else "Limited evidence")

    recs = generate_recommendations(text, job_description, matched_skills=matched,
                                    missing_skills=missing, similarity=sim)
    summary = (f"Resume text supports {supported_count} of {len(requirements)} extracted job criteria. Each supported item includes its closest evidence passage; unsupported items are not assumed to be absent from your experience."
               if requirements else "Could not reliably extract individual criteria from this job description. Review the input and try a fuller posting.")
    ats_checks = [
        {"name": "Contact details", "passed": bool(re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text)), "detail": "Email address found" if re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text) else "Add a clearly labelled email address."},
        {"name": "Readable sections", "passed": len(headings) >= 3, "detail": f"{len(headings)} clearly labeled sections detected" if len(headings) >= 3 else "Use standard headings such as Experience, Skills, and Education."},
        {"name": "Resume length", "passed": 150 <= len(text.split()) <= 1200, "detail": f"{len(text.split())} words extracted"},
        {"name": "Role evidence", "passed": supported_count > 0, "detail": f"{supported_count} of {len(requirements)} extracted criteria have supporting resume text"},
        {"name": "Measurable impact", "passed": quant > 0, "detail": f"Found quantified detail in {quant} resume lines" if quant else "Add truthful numbers to show scale or outcomes."},
    ]
    result = {"score": score, "verdict": verdict, "mismatch": mismatch, "summary": summary,
            "analysis_method": semantic["analysis_method"] + " + criterion-level evidence", "model": model_status(), "breakdown": breakdown, "matched": matched,
            "missing": missing, "extra": extra[:12], "similarity": round(sim * 100),
            "skill_priorities": skill_priorities,
            "words": len(text.split()), "skill_evidence": skill_evidence,
            "requirements": requirements, "requirements_supported": supported_count,
            "requirements_total": len(requirements), "ats_checks": ats_checks, "recommendations": recs}
    with connection() as conn:
        conn.execute(
            "INSERT INTO analyses (filename, job_title, job_description, score, verdict, summary, result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ((resume.filename or "").strip(), (job_description.strip().splitlines()[0][:160] if job_description.strip() else "Role"), job_description.strip(), score, verdict, summary, json.dumps(result)),
        )
    return result


@app.get("/api/history", tags=["Analysis history"])
def analysis_history(limit: int = Query(default=30, ge=1, le=100)):
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, job_title, score, verdict, summary, created_at FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


@app.delete("/api/history/{analysis_id}", status_code=204, tags=["Analysis history"])
def delete_analysis(analysis_id: int):
    with connection() as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        if cursor.rowcount == 0:
            raise HTTPException(404, "Analysis history item not found.")


def store_pdf_preview(data: bytes) -> str:
    global PDF_PREVIEW_BYTES
    token = secrets.token_urlsafe(18)
    while PDF_PREVIEWS and PDF_PREVIEW_BYTES + len(data) > MAX_PREVIEW_BYTES:
        old_token, old_data = next(iter(PDF_PREVIEWS.items()))
        PDF_PREVIEWS.pop(old_token)
        PDF_PREVIEW_BYTES -= len(old_data)
    if len(data) > MAX_PREVIEW_BYTES:
        raise HTTPException(413, "This resume is too large to keep as a temporary preview.")
    PDF_PREVIEWS[token] = data
    PDF_PREVIEW_BYTES += len(data)
    return f"/api/organization/preview/{token}"


@app.get("/api/organization/preview/{token}", include_in_schema=False)
def organization_preview(token: str):
    data = PDF_PREVIEWS.get(token)
    if data is None:
        raise HTTPException(404, "This resume preview has expired. Run the batch again to create a new preview.")
    return Response(content=data, media_type="application/pdf", headers={"Content-Disposition": "inline"})


@app.post("/api/organization/analyze")
async def organization_analyze(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
    shortlist_size: int = Form(5),
):
    """Compare many resumes for one role and return an explainable review queue.

    This endpoint ranks evidence for recruiter review. It does not make an
    employment decision, infer protected traits, or silently discard a file.
    """
    if len(resumes) == 0:
        raise HTTPException(400, "Upload at least one PDF resume.")
    if len(resumes) > 100:
        raise HTTPException(413, "Upload up to 100 resumes per batch.")
    if not 1 <= shortlist_size <= len(resumes):
        raise HTTPException(400, f"Shortlist size must be between 1 and {len(resumes)}.")
    results, errors = [], []
    for position, resume in enumerate(resumes):
        name = resume.filename or f"resume-{position + 1}.pdf"
        try:
            result = await analyze(resume=resume, job_description=job_description)
            # Preserve the source filename and make uncertainty visible in the queue.
            result["filename"] = name
            result["review_required"] = result.get("mismatch", False) or result.get("score", 0) < 45
            result["shortlist_recommended"] = False
            results.append(result)
        except HTTPException as exc:
            errors.append({"filename": name, "error": str(exc.detail)})
    results.sort(key=lambda item: (item.get("score", 0), item.get("requirements_supported", 0)), reverse=True)
    for rank, result in enumerate(results, 1):
        result["rank"] = rank
        result["shortlist_recommended"] = rank <= shortlist_size and not result["review_required"]
    return {
        "job_description_valid": True,
        "candidate_count": len(resumes),
        "analyzed_count": len(results),
        "shortlist_size": shortlist_size,
        "candidates": results,
        "errors": errors,
        "human_review_required": True,
        "notice": "Use this queue as decision support. Review the evidence for every candidate before contacting, advancing, or rejecting anyone.",
    }


@app.get("/api/jobs", tags=["Job tracker"])
def list_jobs(q: str = Query(default="", max_length=120), status: str = Query(default="", max_length=20)):
    if status and status not in JOB_STATUSES:
        raise HTTPException(400, "Unknown job status.")
    with connection() as conn:
        sql = "SELECT * FROM jobs WHERE 1=1"
        args = []
        if q.strip():
            pattern = f"%{q.strip()}%"
            sql += " AND (company LIKE ? OR title LIKE ? OR location LIKE ?)"
            args.extend([pattern, pattern, pattern])
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY updated_at DESC, id DESC"
        return [dict(row) for row in conn.execute(sql, args).fetchall()]


@app.post("/api/jobs", status_code=201, tags=["Job tracker"])
def create_job(payload: JobCreate):
    url = payload.url.strip()
    if url and not url.startswith(("https://", "http://")):
        raise HTTPException(400, "Job URL must start with http:// or https://.")
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO jobs (company, title, location, url, description) VALUES (?, ?, ?, ?, ?)",
            (payload.company.strip(), payload.title.strip(), payload.location.strip(), url, payload.description.strip()),
        )
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)


@app.patch("/api/jobs/{job_id}", tags=["Job tracker"])
def update_job(job_id: int, payload: JobUpdate):
    with connection() as conn:
        cursor = conn.execute("UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (payload.status, job_id))
        if cursor.rowcount == 0:
            raise HTTPException(404, "Saved job not found.")
        return dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


@app.delete("/api/jobs/{job_id}", status_code=204, tags=["Job tracker"])
def delete_job(job_id: int):
    with connection() as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        if cursor.rowcount == 0:
            raise HTTPException(404, "Saved job not found.")
