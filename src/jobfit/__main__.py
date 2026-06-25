import sys
from .function_caller import run_tool_workflow

SAMPLE_CONTEXT = {
    "resume_path": "data/resume_sentences.txt",
    "jd_path": "data/jd_sentences.txt",
    "jd_text": "We are looking for an AI Engineer with experience in Python, Machine Learning, Docker, FastAPI and MLOps. The candidate should have experience deploying models and building scalable ML systems.",
    "resume_text": "Experienced software engineer skilled in Java, Python, C++, and data structures. Strong background in building scalable systems and leading engineering teams.",
}

if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
else:
    query = "How well does my resume match this job?"

out = run_tool_workflow(query, context=SAMPLE_CONTEXT, verbose=True)
answer = out["answer"]
print(f"\nAnswer: {answer}")
