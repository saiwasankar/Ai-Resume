# RESUME ANALYZER 1

Upload your resume as a PDF, paste a job description, and get an instant analysis: an overall score out of 100, the skills you already match, the skills you're missing, and specific recommendations to improve your resume.

![Screenshot](screenshots/home.png)
![Results](screenshots/results.png)

## Features

- PDF resume upload with drag and drop
- Overall score out of 100 with a category breakdown
- Matched and missing skills compared against the job description
- Strengths, areas to improve, and actionable recommendations
- Runs fully offline with no paid APIs

## How the score works

| Category | Points |
|---|---|
| Skills match | 40 |
| Keyword relevance | 20 |
| Experience | 20 |
| Quantified achievements | 10 |
| Resume structure | 10 |

## Tech stack

- **Backend:** Python, FastAPI, pdfplumber
- **Frontend:** HTML, CSS, JavaScript

## Project structure

```
resume-analyzer/
├── app/
│   ├── main.py        # API endpoint and server
│   ├── parser.py      # PDF to text
│   ├── skills.py      # skill extraction and matching
│   ├── scorer.py      # 100-point scoring
│   └── feedback.py    # strengths, improvements, recommendations
├── frontend/
│   └── index.html     # user interface
├── requirements.txt
└── README.md
```

## Getting started

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/resume-analyzer-1.git
cd resume-analyzer-1

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## Usage

1. Upload a text-based PDF resume (exported from Word or Google Docs).
2. Paste the full job description.
3. Click **Analyze resume** to see your score and feedback.

Scanned or image-only PDFs aren't supported yet.

## Roadmap

- [ ] OCR support for scanned PDFs
- [ ] Larger skills database for more job roles
- [ ] Optional AI-written feedback
- [ ] Downloadable report

## Contributing

Suggestions and pull requests are welcome. Open an issue to discuss what you'd like to change.

## Author

Built by **PARTH SHAILESH SURYARAO**. Connect with me on [LinkedIn](https://www.linkedin.com/in/parth-suryarao-342875315/).