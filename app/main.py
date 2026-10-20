"""RoleFit API and web entry point. Interactive OpenAPI docs are served at /docs."""
import io
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


def extract_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


@app.get("/")
def home():
    return FileResponse(INDEX)


@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "service": "RoleFit", "version": app.version}


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
    return {"score": score, "verdict": verdict, "mismatch": mismatch, "summary": summary,
            "analysis_method": semantic["analysis_method"] + " + criterion-level evidence", "breakdown": breakdown, "matched": matched,
            "missing": missing, "extra": extra[:12], "similarity": round(sim * 100),
            "words": len(text.split()), "skill_evidence": skill_evidence,
            "requirements": requirements, "requirements_supported": supported_count,
            "requirements_total": len(requirements), "ats_checks": ats_checks, "recommendations": recs}


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
