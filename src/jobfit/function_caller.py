import json
import re
from .llm import generate_text
from .tools import calculate_match_score, extract_skills, summarize_jd_llm
from .validator import validate_json


# ── Tool schemas (for documentation/future use) ───────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "calculate_match_score",
        "description": "Calculates how well a resume matches a job description. Use when user asks about fit, match, or compatibility.",
        "parameters": {
            "type": "object",
            "properties": {
                "resume_sentences": {"type": "array",  "description": "List of resume sentences"},
                "jd_sentences":     {"type": "array",  "description": "List of JD sentences"}
            },
            "required": ["resume_sentences", "jd_sentences"]
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

# ── Routing schema — only name + description sent to LLM ─────────────────────
ROUTING_SCHEMAS = [
    {"name": schema["name"], "description": schema["description"]}
    for schema in TOOL_SCHEMAS
]

TOOL_REGISTRY = {
    "calculate_match_score": calculate_match_score,
    "summarize_jd":          summarize_jd_llm,
    "extract_skills":        extract_skills,
}

TOOL_CALL_SCHEMA = {"name": str}
RESULT_SCHEMAS = {
    "calculate_match_score": {"score": float, "top_matches": list},
}


# ── Text splitter ─────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list:
    """Split raw text into sentences for embedding."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        raise ValueError("Text is empty — cannot split into sentences.")
    return sentences


# ── Parser ────────────────────────────────────────────────────────────────────
def parse_tool_call(response: str):
    """Expects LLM to return: {"name": "tool_name"}"""
    cleaned = re.sub(r"```json|```", "", response).strip()
    match   = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        if "name" in parsed and parsed["name"] in TOOL_REGISTRY:
            return {"name": parsed["name"]}
        return None
    except json.JSONDecodeError:
        return None


# ── Keyword fallback ──────────────────────────────────────────────────────────
_KEYWORD_MAP = {
    "calculate_match_score": {"match", "compare", "fit", "compatible"},
    "summarize_jd":          {"summarize", "summary", "overview", "summarise"},
    "extract_skills":        {"skill", "extract"},
}

def _keyword_fallback(query: str):
    q = query.lower()
    for tool_name, keywords in _KEYWORD_MAP.items():
        if any(kw in q for kw in keywords):
            return tool_name
    return None


# ── Main agent loop ───────────────────────────────────────────────────────────
def run_tool_workflow(user_query: str, context: dict = None, verbose: bool = True):
    """
    context keys:
        resume_text  → raw resume text (used by calculate_match_score + extract_skills)
        jd_text      → raw JD text    (used by calculate_match_score + summarize_jd)
    """
    context = context or {}

    if verbose:
        print(f"\n{'='*50}")
        print(f"USER: {user_query}")
        print(f"{'='*50}")

    routing_context = json.dumps(ROUTING_SCHEMAS, indent=2)

    decision_prompt = f"""You have access to these tools:
{routing_context}

RULES:
- If a tool matches the user's intent, respond ONLY with JSON in this exact format:
{{"name": "tool_name"}}
- Replace tool_name with one of: calculate_match_score, summarize_jd, extract_skills
- If no tool matches, reply in plain text directly.
- No explanation. No markdown. No extra keys.

User: {user_query}
"""

    llm_output = generate_text(decision_prompt, temperature=0.1, max_new_tokens=50)

    if verbose:
        print(f"\n[LLM Raw] {llm_output}")

    tool_call = parse_tool_call(llm_output)
    routing   = "llm"

    if tool_call and not validate_json(tool_call, TOOL_CALL_SCHEMA):
        if verbose:
            print("[Validate] Malformed tool call — falling back")
        tool_call = None

    if not tool_call:
        tool_name = _keyword_fallback(user_query)
        if tool_name:
            routing = "keyword"
            if verbose:
                print(f"[Fallback] Keyword matched → {tool_name}")
        else:
            direct_answer = generate_text(user_query, temperature=0.3, max_new_tokens=200)
            return {"answer": direct_answer, "tool": None, "routing": "direct"}
    else:
        tool_name = tool_call["name"]

    if verbose:
        print(f"[Tool] {tool_name} | Route: {routing}")

    if tool_name not in TOOL_REGISTRY:
        return {"answer": f"Unknown tool: {tool_name}", "tool": tool_name, "routing": routing}

    # Build parameters from text context
    if tool_name == "calculate_match_score":
        params = {
            "resume_sentences": _split_sentences(context.get("resume_text", "")),
            "jd_sentences":     _split_sentences(context.get("jd_text", ""))
        }
    elif tool_name == "summarize_jd":
        params = {"jd_text": context.get("jd_text", "")}
    elif tool_name == "extract_skills":
        text   = context.get("resume_text") or context.get("jd_text") or ""
        params = {"text": text}

    try:
        result = TOOL_REGISTRY[tool_name](**params)
        if verbose:
            print(f"[Result] {result}")
    except Exception as e:
        return {"answer": f"Tool error: {e}", "tool": tool_name, "routing": routing}

    result_schema = RESULT_SCHEMAS.get(tool_name)
    if result_schema and isinstance(result, dict) and not validate_json(result, result_schema):
        return {"answer": "Tool returned data in wrong format.", "tool": tool_name, "routing": routing}

    if isinstance(result, str):
        final_answer = result
    else:
        final_prompt = f"""The user asked: "{user_query}"
Tool used: {tool_name}
Result: {result}

Give a clear, short final answer based on the result."""
        final_answer = generate_text(final_prompt, temperature=0.3, max_new_tokens=150)

    if verbose:
        print(f"\n[Final] {final_answer}")

    return {"answer": final_answer, "tool": tool_name, "routing": routing}


if __name__ == "__main__":
    sample_context = {
        "resume_text": "Experienced software engineer skilled in Python, C++, and data structures. Strong background in building scalable systems.",
        "jd_text":     "We are looking for an AI Engineer with experience in Python, Machine Learning, Docker, FastAPI and MLOps.",
    }
    tests = [
        ("How well does my resume match this job?", sample_context),
        ("Summarize this job description for me",   {"jd_text": sample_context["jd_text"]}),
        ("What skills are in this resume?",         {"resume_text": sample_context["resume_text"]}),
    ]
    for q, ctx in tests:
        out = run_tool_workflow(q, context=ctx, verbose=True)