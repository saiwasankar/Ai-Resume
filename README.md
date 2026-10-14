# RoleFit — Resume Analyzer

RoleFit compares a text-based PDF resume with a target job description. The workflow starts with the job description, then asks for the resume, and returns role alignment, supported and missing skills, and recommendations based on the supplied text.

## Analysis approach

- Uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` encoder for semantic text similarity when the model can be loaded.
- Falls back to TF-IDF if sentence-transformers is unavailable or the model cannot be loaded. The response names the method actually used.
- Checks role skills and resume evidence separately; semantic similarity alone does not establish that a candidate has a skill.
- Flags very low relevance with no shared role skills as a likely mismatch and caps that score at 20.

This project does **not** train on a real-time resume database, scrape resumes, or retain uploaded resumes as training data. The encoder is a general pretrained language model. It downloads on first use and requires internet access then. No private or live industry resume dataset is configured. The score is an aid for reviewing alignment, not a hiring decision.

## Run locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Upload a text-based PDF (up to 5 MB). Scanned/image PDFs are not supported without OCR.

## Stack

- FastAPI, pdfplumber
- HTML, CSS, JavaScript
- sentence-transformers and scikit-learn
