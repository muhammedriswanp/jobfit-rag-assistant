from .function_caller import run_tool_workflow

resume_text = """Experienced software engineer skilled in Python, C++ and data structures.
Built and deployed ML models using scikit-learn and PyTorch.
Used Docker and FastAPI to containerize and serve REST APIs."""

jd_text = """We are looking for an AI Engineer with experience in Python and Machine Learning.
Candidate must know Docker, FastAPI and MLOps tools.
Experience with model deployment and monitoring is required."""

tests = [
    "How well does my resume match this job?",
    "Summarize this job description for me",
    "What skills are in this resume?",
]

for q in tests:
    run_tool_workflow(q, resume_text=resume_text, jd_text=jd_text, verbose=True)