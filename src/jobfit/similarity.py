from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_sentences, jd_sentences):

    embedding_model  = SentenceTransformer("all-MiniLM-L6-v2")

    resume_embeddings = embedding_model.encode(resume_sentences)
    jd_embeddings = embedding_model.encode(jd_sentences)

    similarity_matrix = cosine_similarity(jd_embeddings, resume_embeddings)

    return similarity_matrix

def get_best_matches(resume_sentences, jd_sentences):

    matches = []

    similarity_matrix = compute_similarity(
        resume_sentences,
        jd_sentences
    )

    for i, jd in enumerate(jd_sentences):

        best_index = similarity_matrix[i].argmax()

        best_score = similarity_matrix[i][best_index]

        best_resume = resume_sentences[best_index]

        matches.append({
            "jd": jd,
            "resume": best_resume,
            "score": float(best_score)
        })

    return matches

if __name__ == "__main__":

    resume_file = "data/resume_sentences.txt"
    jd_file = "data/jd_sentences.txt"

    with open(resume_file, "r", encoding="utf-8") as f:
        resume_sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    with open(jd_file, "r", encoding="utf-8") as f:
        jd_sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    matches = get_best_matches(
    resume_sentences,
    jd_sentences
)

    for match in matches:
        print("=" * 50)
        print("JD:")
        print(match["jd"])

        print("\nBest Resume Match:")
        print(match["resume"])

        print("\nSimilarity Score:")
        print(f"{match['score']:.3f}")

        print("=" * 50)
