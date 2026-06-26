from .similarity import get_best_matches, SIMILARITY_THRESHOLD
from .llm import generate_text


def calculate_match_score(resume_sentences, jd_sentences):
    matches = get_best_matches(resume_sentences, jd_sentences)
    avg_score = sum(m["score"] for m in matches) / len(matches) if matches else 0
    top = [m for m in matches if m["score"] >= SIMILARITY_THRESHOLD]
    return {"score": round(avg_score, 3), "top_matches": top}


def extract_skills(text):
    prompt = (
        "Extract all technical and professional skills from the text below. "
        "Return them as a comma-separated list. No explanation.\n\n"
        f"Text:\n{text}\n\nSkills:"
    )
    result = generate_text(
        prompt,
        system_prompt="You are a skill extractor. Return only a comma-separated list. No explanation. No extra text.",
        temperature=0.2,
        max_new_tokens=100
    )
    skills = [s.strip().strip("[]\"'") for s in result.replace("\n", ",").split(",") if s.strip()]
    return skills if skills else []


def summarize_jd_llm(jd_text):
    prompt = (
        "Summarize this job description in 2-3 sentences. "
        "Focus on role, key responsibilities, and required skills.\n\n"
        f"Job Description:\n{jd_text}\n\nSummary:"
    )
    return generate_text(
        prompt,
        system_prompt="You are a job description summarizer. Be concise and factual. No extra commentary.",
        temperature=0.3,
        max_new_tokens=150
    )