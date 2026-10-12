

from pathlib import Path
 
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
 
from app.parser import extract_text_from_pdf
from app.skills import compare_skills
from app.scorer import calculate_score
from app.feedback import generate_feedback
 
app = FastAPI(title="RESUME ANALYZER 1")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Absolute path to the frontend folder, so it works no matter where you run uvicorn from
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
 
 
@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
 
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Please paste a job description")
 
    try:
        resume_text = extract_text_from_pdf(resume.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open this PDF. Is the file valid?")
 
    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not read text from this PDF (is it scanned?)",
        )
 
    skills = compare_skills(resume_text, job_description)
    scoring = calculate_score(resume_text, job_description, skills)
    feedback = generate_feedback(resume_text, job_description, skills, scoring)
 
    return {**scoring, "skills": skills, **feedback}
 
 
# Keep this LAST so it doesn't override /analyze
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
 

