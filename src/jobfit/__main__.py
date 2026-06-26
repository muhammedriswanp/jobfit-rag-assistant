import sys
from .function_caller import run_tool_workflow

SAMPLE_CONTEXT = {
    "resume_text": "Experienced software engineer skilled in Python, C++, and data structures. Strong background in building scalable systems.",
    "jd_text":     "We are looking for an AI Engineer with experience in Python, Machine Learning, Docker, FastAPI and MLOps. The candidate should have experience deploying models and building scalable ML systems.",
}

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How well does my resume match this job?"

out = run_tool_workflow(query, context=SAMPLE_CONTEXT, verbose=False)
print(f"\nAnswer: {out['answer']}")