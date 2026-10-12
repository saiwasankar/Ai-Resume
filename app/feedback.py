import re


def generate_feedback(resume_text: str, jd_text: str, skills: dict, scoring: dict) -> dict:
    t = resume_text.lower()
    parts = {b["label"]: b["score"] / b["max"] for b in scoring["breakdown"]}
    matched, missing = skills["matched"], skills["missing"]

    strengths, improvements, recs = [], [], []

    # ---- Skills ----
    if matched:
        strengths.append(
            f"Your resume already shows {len(matched)} skills this job asks for: "
            f"{', '.join(matched[:6])}."
        )
    if missing:
        improvements.append(
            f"{len(missing)} skills from the job description are missing: "
            f"{', '.join(missing[:6])}."
        )
        recs.append(
            f"If you have any experience with {', '.join(missing[:3])}, add it to your "
            "Skills section and to a project or job bullet that shows how you used it."
        )
        recs.append(
            f"For skills you haven't used yet, build a small project with "
            f"{missing[0]} and list it under Projects."
        )
    elif matched:
        strengths.append("You cover every skill from the job description that we could detect.")

    # ---- Keywords ----
    if parts["Keyword relevance"] >= 0.6:
        strengths.append("Your wording closely matches the language of the job description.")
    else:
        improvements.append("Your resume uses little of the job description's own wording.")
        recs.append(
            "Reuse exact terms from the job posting (tools, job titles, responsibilities) "
            "where they honestly apply. Many companies filter resumes by keyword."
        )

    # ---- Experience ----
    if parts["Experience"] >= 0.7:
        strengths.append("Your experience section is clear and uses strong action verbs.")
    else:
        improvements.append("Experience bullets are weak on action verbs or missing dates.")
        recs.append(
            "Start each bullet with a verb such as Built, Led, Optimized or Delivered, "
            "and include dates for every role or project."
        )

    # ---- Quantified achievements ----
    if parts["Quantified achievements"] >= 0.6:
        strengths.append("You back up your work with numbers, which recruiters look for.")
    else:
        improvements.append("Few or no measurable results in your bullets.")
        recs.append(
            "Add numbers to your bullets. Example: 'Built a REST API used by 500+ users, "
            "cutting response time by 30%'."
        )

    # ---- Structure ----
    if parts["Resume structure"] >= 0.7:
        strengths.append("Your resume has the standard sections and a good length.")
    else:
        improvements.append("Structure or length needs attention.")
        recs.append(
            "Include clear sections for Summary, Experience, Education, Skills and Projects, "
            "and keep the resume to 1 or 2 pages."
        )

    # ---- Contact details ----
    has_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text)
    has_phone = re.search(r"\+?\d[\d\s().-]{8,}\d", resume_text)
    if not (has_email and has_phone):
        improvements.append("Email or phone number could not be found.")
        recs.append("Put your email and phone number at the top of the resume as plain text.")
    if "linkedin" not in t and "github" not in t:
        recs.append("Add your LinkedIn or GitHub link so recruiters can see more of your work.")

    return {
        "strengths": strengths[:5],
        "improvements": improvements[:5],
        "recommendations": recs[:7],
    }