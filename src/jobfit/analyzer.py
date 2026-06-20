from similarity import get_best_matches
from qa import answer_question
from summarizer import summarize_jd

def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
def analyze(resume_path, jd_path):
    resume_sentences = load_sentences(resume_path)
    jd_sentences = load_sentences(jd_path)

    matches = get_best_matches(resume_sentences, jd_sentences)
    print("=" * 50)
    print("TOP RESUME MATCHES")
    print("=" * 50)

    for match in matches:
        print(f"\nJD: {match['jd']}")
        print(f"Best Resume Match: {match['resume']}")
        print(f"Score: {match['score'] * 100:.1f}%")
    
    jd_text = "\n".join(jd_sentences)

    summary = summarize_jd(jd_text)
    print("\n" + "=" * 50)
    print("JD SUMMARY")
    print("=" * 50)

    print(summary)

    question = "What skills are required?"
    answer = answer_question(jd_text, question)

    print("\n" + "=" * 50)
    print("QUESTION ANSWERING")
    print("=" * 50)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    analyze("data/resume_sentences.txt", "data/jd_sentences.txt")
