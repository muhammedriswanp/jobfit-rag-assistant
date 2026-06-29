from .similarity import get_best_matches, SIMILARITY_THRESHOLD
from .llm import generate_text
from .vector_store import retrieve, resume_collection, jd_collection

def calculate_match_score(query: str = "skills experience projects"):
    resume_chunks = retrieve(query, resume_collection, k=3)
    jd_chunks = retrieve(query, jd_collection, k=3)

    
    matches = get_best_matches(resume_chunks, jd_chunks)
    avg_score = sum(m["score"] for m in matches) / len(matches) if matches else 0
    top = [m for m in matches if m["score"] >= SIMILARITY_THRESHOLD]
    return {"score": round(avg_score, 3), "top_matches": top}


def extract_skills(query: str="technical skills experience tools"):
    resume_chunks = retrieve(query, resume_collection, k=3)
    text= " ".join(resume_chunks)
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


def summarize_jd_llm(query: str= "role responsibilities requirements skills"):
    jd_chunks = retrieve(query, jd_collection, k=3)
    jd_text = " ".join(jd_chunks)
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