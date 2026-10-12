# RESUME ANALYZER 1

Resume Analyzer is a web-based application that analyzes your resume against a target job description. Upload your resume as a PDF to receive an overall score out of 100, matched and missing skills, text similarity insights, and actionable recommendations to improve your resume.

![Home Page](screenshots/home.png)

![Analysis Results](screenshots/results.png)

## Features

- **PDF Resume Upload:** Upload your resume using file browsing or drag and drop.
- **Resume Match Score:** Get an overall score out of 100.
- **Score Breakdown:** View your performance across different resume evaluation categories.
- **Skill Gap Analysis:** Identify skills that match the job description and skills that may be missing.
- **Text Similarity:** Compare resume content with the job description using TF-IDF text similarity.
- **Personalized Feedback:** Receive recommendations to improve resume relevance.
- **Modern User Interface:** Responsive lavender and purple-themed dashboard.
- **Fast Analysis:** Analyze your resume and view results through a simple web interface.

## How the Score Works

The application calculates a resume score using multiple evaluation categories. The exact weights depend on the scoring configuration implemented in `app/scorer.py`.

The evaluation can include:

- Skills match
- Keyword relevance
- Text similarity
- Relevant experience
- Quantified achievements
- Resume structure

The final score is intended to help users understand how closely their resume aligns with a particular job description.

## Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **PDF Processing:** pdfplumber
- **Text Similarity:** scikit-learn and TF-IDF
- **API Communication:** REST API using JSON responses and multipart form data

## Project Structure

```text
resume-analyzer/
├── app/
│   ├── main.py          # API endpoint and application setup
│   ├── parser.py        # Extracts text from PDF resumes
│   ├── skills.py        # Skill extraction and matching
│   ├── scorer.py        # Calculates the resume score
│   ├── feedback.py      # Generates feedback and recommendations
│   └── semantic.py      # Text similarity comparison
├── frontend/
│   └── index.html       # Responsive web interface
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Getting Started

### Requirements

- Python installed on your system
- Git
- A modern web browser

### 1. Clone the Repository

Replace the example URL with your actual GitHub repository URL.

```bash
git clone https://github.com/suryaraoparth35/RESUME-ANALYZER-1
cd resume-analyzer-1
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Application

```bash
python -m uvicorn app.main:app --reload
```

### 5. Open the Application

Visit the following address in your browser:

http://127.0.0.1:8000

## Usage

1. Open the application in your browser.
2. Upload your resume in PDF format.
3. Paste the job description for the position you are targeting.
4. Click **Analyze my resume**.
5. Review your overall score, skills match, text similarity, and recommendations.
6. Improve your resume based on the feedback and analyze it again.

**Note:** The application currently supports text-based PDF resumes. Scanned or image-only PDFs may not be readable without OCR support.

## Roadmap

- [ ] OCR support for scanned PDF resumes
- [ ] Expand the skills database for different job roles
- [ ] Improve semantic similarity beyond TF-IDF
- [ ] More detailed, role-specific recommendations
- [ ] Downloadable analysis reports
- [ ] Additional resume evaluation metrics

## Contributing

Suggestions, bug reports, and contributions are welcome. Feel free to open an issue or submit a pull request to help improve the project.

## Author

**Parth Shailesh Suryarao**

Connect with me on [LinkedIn](https://www.linkedin.com/in/parth-suryarao-342875315/).

---

