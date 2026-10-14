# RoleFit — Resume Intelligence

RoleFit is a local-first resume analysis and job application workspace. It compares a text-based PDF resume with a pasted job description, reports skill and ATS evidence, and keeps a private job application tracker in SQLite.

## Start and stop the local server

From Terminal, open this project folder and run:

```bash
cd "/Users/saiwasankar/Documents/resume analyser"
./start_server.sh
```

Open <http://127.0.0.1:8000>. Interactive API documentation is at <http://127.0.0.1:8000/docs> and health status is at <http://127.0.0.1:8000/api/health>.

The first run creates `.venv` and installs the small backend requirements. The start command keeps the server in the terminal and logs to `server.log`. Press **Ctrl+C** there, or open another Terminal in this folder and run:

```bash
./stop_server.sh
```

You can also double-click `start_server.command` on macOS, then `stop_server.command` to stop it. To use a different local port, set it before starting, e.g. `PORT=8010 ./start_server.sh`.

## Backend capabilities

- **Resume analysis:** `/api/analyze` (also available as `/analyze`) checks PDF readability, role skill overlap, semantic/text relevance, experience, impact and structure. It rejects oversized or malformed PDFs and flags likely role mismatches.
- **ATS readiness checks:** contact details, recognizable sections, text length, role evidence and quantified impact.
- **Input quality gates:** rejects text that does not look like a job description or a structured resume, and returns an error instead of a score for clearly unrelated pairs.
- **Skill labels stay specific:** only recognized skills are listed as skills; generic or frequent job-description words remain ordinary text.
- **Evidence-linked feedback:** recommendations are drawn from the job description and resume text; matched skills include their source resume lines in the API response.
- **Local saved job tracker:** add, search, filter, update application status, and delete saved opportunities. Records live in `data/rolefit.sqlite3`; resume files and extracted resume text are not stored.
- **API operations:** `/api/health`, `/api/jobs` CRUD, plus interactive OpenAPI/Swagger docs at `/docs`.

The job tracker is local and user populated; this project does not currently connect to a live job feed, account system, AI chat provider, or community backend. Uploaded resumes are not training data. The sentence-transformers encoder is optional; install it separately with `pip install -r requirements-ai.txt`. Without it, RoleFit reports and uses its TF-IDF fallback.

## Requirements

Python 3.9+ is recommended. Core dependencies: FastAPI, Uvicorn, pdfplumber and scikit-learn. Resume PDFs must contain selectable text; scanned/image-only PDFs need OCR, which is not currently included. The local analyzer is a decision aid, not a hiring decision.
