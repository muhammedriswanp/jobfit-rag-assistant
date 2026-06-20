from .similarity import get_best_matches, SIMILARITY_THRESHOLD
from .qa import answer_question
from .summarizer import summarize_jd

def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def analyze(resume_path, jd_path):
    resume_sentences = load_sentences(resume_path)
    if not resume_sentences:
        raise ValueError("Resume file is empty.")
    
    jd_sentences = load_sentences(jd_path)
    if not jd_sentences:
        raise ValueError("Job discription file is empty.")


    matches = get_best_matches(resume_sentences, jd_sentences)
    print("=" * 50)
    print("TOP RESUME MATCHES")
    print("=" * 50)

    filtered_matches = [
        match
        for match in matches
        if match['score'] >= SIMILARITY_THRESHOLD
    ]

    for match in filtered_matches:
        print(f"\nJD: {match['jd']}")
        print(f"Best Resume Match: {match['resume']}")
        print(f"Score: {match['score'] * 100:.1f}%")
    
    jd_text = "\n".join(jd_sentences)

    summary = summarize_jd(jd_text)
    print("\n" + "=" * 50)
    print("JD SUMMARY")
    print("=" * 50)

    print(summary)

    questions = [
    "What is the job title?",
    "What experience is required?",
    "What tools are mentioned?",
    "What technical skills are mandatory?"]

    for question in questions:        
        answer = answer_question(jd_text, question)

        print("\n" + "=" * 50)
        print("QUESTION ANSWERING")
        print("=" * 50)

        print(f"Question: {question}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    analyze("data/resume_sentences.txt", "data/jd_sentences.txt")
