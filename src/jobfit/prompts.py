import json
from .llm import generate_text


def parse_json_safe(raw: str) -> dict:
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"role": "", "skills": [], "experience": "", "tools": []}


def extract_skills_prompt(jd_text: str) -> str:
    return f"""Extract structured information from a job description and return ONLY valid JSON.
No explanation. No markdown. No extra text. Just the JSON object.

### EXAMPLE 1
Job Description:
We are looking for a Data Analyst with 2+ years of experience.
Required skills: Python, SQL, Tableau. Familiarity with AWS is a plus.

Output:
{{"role": "Data Analyst", "skills": ["Python", "SQL", "Tableau"], "experience": "2+ years", "tools": ["AWS"]}}

### EXAMPLE 2
Job Description:
Hiring a Machine Learning Engineer to build and deploy ML models.
Must know: PyTorch, Scikit-learn, Docker, MLflow. 3-5 years required.

Output:
{{"role": "Machine Learning Engineer", "skills": ["PyTorch", "Scikit-learn"], "experience": "3-5 years", "tools": ["Docker", "MLflow"]}}

### NOW EXTRACT
Job Description:
{jd_text}

Output:"""


def summarize_role_prompt(jd_text: str) -> str:
    return f"""Summarize this job in exactly 2 sentences.
Sentence 1: role and main responsibility.
Sentence 2: most important skill required.
No bullet points. No extra text.

Job: {jd_text}

Two sentence summary:"""


def gap_analysis_prompt(resume_text: str, jd_text: str) -> str:
    return f"""You are a career coach reviewing a resume against a job description.
Identify skills or experience that the job requires but are MISSING or WEAK in the resume.

Rules:
- Only mention gaps that are clearly required in the JD
- Do not mention skills already present in the resume
- Return 3 to 5 bullet points maximum
- Each bullet: one short sentence, start with a dash (-)
- No explanation, no intro, no extra text

Resume:
{resume_text}

Job Description:
{jd_text}

Gaps:"""


def extract_structured_jd(jd_text: str) -> dict:
    prompt = extract_skills_prompt(jd_text)
    raw_output = generate_text(prompt, temperature=0.2)
    return parse_json_safe(raw_output)
