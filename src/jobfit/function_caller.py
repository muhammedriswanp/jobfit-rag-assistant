import json
import re
from .llm import generate_text
from .tools import calculate_match_score, extract_skills, summarize_jd_llm


TOOL_SCHEMAS = [
    {
        "name": "calculate_match_score",
        "description": "Calculates how well a resume matches a job description. Use when user asks about fit, match, or compatibility.",
        "parameters": {
            "type": "object",
            "properties": {
                "resume_path": {"type": "string", "description": "Path to resume file"},
                "jd_path": {"type": "string", "description": "Path to job description file"}
            },
            "required": ["resume_path", "jd_path"]
        }
    },
    {
        "name": "summarize_jd",
        "description": "Summarizes a job description into key points. Use when user wants a quick understanding of a JD.",
        "parameters": {
            "type": "object",
            "properties": {
                "jd_text": {"type": "string", "description": "Raw job description text"}
            },
            "required": ["jd_text"]
        }
    },
    {
        "name": "extract_skills",
        "description": "Extracts skills from any text (resume or JD). Use when user asks what skills are present or required.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract skills from"}
            },
            "required": ["text"]
        }
    }
]


TOOL_REGISTRY = {
    "calculate_match_score": calculate_match_score,
    "summarize_jd": summarize_jd_llm,
    "extract_skills": extract_skills,
}


def parse_tool_call(response):
    cleaned = re.sub(r"```json|```", "", response).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        if "tool" in parsed:
            return {"name": parsed["tool"], "arguments": parsed.get("parameters", parsed.get("arguments", {}))}
        if "name" in parsed:
            args = parsed.get("arguments", parsed.get("parameters", {}))
            return {"name": parsed["name"], "arguments": args}
        return None
    except json.JSONDecodeError:
        return None


_ALL_TOOL_KEYWORDS = {"match", "compare", "fit", "compatible",
                      "summarize", "summary", "overview", "summarise",
                      "skill", "extract"}


def _keyword_fallback(query):
    q = query.lower()
    if any(w in q for w in ["match", "compare", "fit", "compatible"]):
        return "calculate_match_score"
    if any(w in q for w in ["summarize", "summary", "overview", "summarise"]):
        return "summarize_jd"
    if any(w in q for w in ["skill", "extract"]):
        return "extract_skills"
    return None


def run_tool_workflow(user_query, context=None, verbose=True):
    """
    context: dict with available data
        - resume_path: str  (for calculate_match_score)
        - jd_path: str      (for calculate_match_score)
        - jd_text: str      (for summarize_jd)
        - resume_text: str  (for extract_skills)
    """
    context = context or {}

    if verbose:
        print(f"\n{'='*50}")
        print(f"USER: {user_query}")
        print(f"{'='*50}")

    tool_context = json.dumps(TOOL_SCHEMAS, indent=2)
    decision_prompt = f"""You have access to these tools:
{tool_context}

RULES:
- If a tool matches the user's intent, respond ONLY with JSON.
- Use exactly this format:
{{"tool": "tool_name"}}
- Only pick the tool name, NOT the parameter values.
- If no tool is needed, answer the user directly in plain text.
- No explanation. No markdown.

User: {user_query}
"""

    llm_output = generate_text(decision_prompt, temperature=0.1, max_new_tokens=200)
    if verbose:
        print(f"\n[LLM] {llm_output}")

    tool_call = parse_tool_call(llm_output)
    routing = "llm"

    if not tool_call:
        tool_name = _keyword_fallback(user_query)
        if tool_name:
            routing = "keyword"
            if verbose:
                print(f"\n[Fallback] {tool_name}")
        else:
            direct_answer = generate_text(user_query, temperature=0.3, max_new_tokens=200)
            return {"answer": direct_answer, "tool": None, "routing": "direct"}
    else:
        tool_name = tool_call["name"]

    if verbose:
        print(f"\n[Tool] {tool_name} | Route: {routing}")

    if tool_name not in TOOL_REGISTRY:
        return {"answer": f"Unknown tool: {tool_name}", "tool": tool_name, "routing": routing}

    # Validate: LLM-picked tool should match query keywords
    if routing == "llm" and not any(kw in user_query.lower() for kw in _ALL_TOOL_KEYWORDS):
        if verbose:
            print(f"[Validate] No tool keywords in query — answering directly")
        direct_answer = generate_text(user_query, temperature=0.3, max_new_tokens=200)
        return {"answer": direct_answer, "tool": None, "routing": "direct"}

    # Build real parameters from caller-provided context
    if tool_name == "calculate_match_score":
        params = {"resume_path": context.get("resume_path"), "jd_path": context.get("jd_path")}
    elif tool_name == "summarize_jd":
        params = {"jd_text": context.get("jd_text", "")}
    elif tool_name == "extract_skills":
        text = context.get("resume_text") or context.get("jd_text") or ""
        params = {"text": text}
    else:
        params = {}

    try:
        result = TOOL_REGISTRY[tool_name](**params)
        if verbose:
            print(f"[Result] {result}")
    except Exception as e:
        return {"answer": f"Error: {e}", "tool": tool_name, "routing": routing}

    if isinstance(result, str):
        final_answer = result
    else:
        final_prompt = f"""The user asked: "{user_query}"
You used tool: {tool_name}
Parameters: {params}
Result: {result}

Give a clear, short final answer based on the result."""

        final_answer = generate_text(final_prompt, temperature=0.3, max_new_tokens=150)

    if verbose:
        print(f"\n[Final] {final_answer}")

    return {"answer": final_answer, "tool": tool_name, "routing": routing}


if __name__ == "__main__":
    sample_context = {
        "resume_path": "data/resume_sentences.txt",
        "jd_path": "data/jd_sentences.txt",
        "jd_text": "We are looking for an AI Engineer with experience in Python, Machine Learning, Docker, FastAPI and MLOps. The candidate should have experience deploying models and building scalable ML systems.",
        "resume_text": "Experienced software engineer skilled in Java, Python, C++, and data structures. Strong background in building scalable systems and leading engineering teams.",
    }
    tests = [
        ("How well does my resume match this job?", sample_context),
        ("Summarize this job description for me", {"jd_text": sample_context["jd_text"]}),
        ("What skills are in this resume?", {"resume_text": sample_context["resume_text"]}),
    ]
    for q, ctx in tests:
        out = run_tool_workflow(q, context=ctx, verbose=True)
        print(f"\nResult: {out['answer']}\n")
