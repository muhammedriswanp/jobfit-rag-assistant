import sys 
from .function_caller import run_tool_workflow
from .ingestion import load_pdf

resume_path = sys.argv[1] if len(sys.argv) > 1 else "data/resume.pdf"
jd_path = sys.argv[2] if len(sys.argv) > 2 else "data/jd.pdf"

resume_chunks = load_pdf(resume_path)
jd_chunks     = load_pdf(jd_path)

print(f"Resume: {len(resume_chunks)} chunks")
print(f"JD:     {len(jd_chunks)} chunks")

tests = [
    "How well does my resume match this job?",
    "Summarize this job description for me",
    "What skills are in this resume?",
]

for q in tests:
    run_tool_workflow(q, resume_chunks=resume_chunks, jd_chunks=jd_chunks, verbose=True)