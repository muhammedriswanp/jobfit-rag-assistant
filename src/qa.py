import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

def answer_question(context, question):

    inputs = tokenizer(question, context, return_tensors="pt")

    output  = model(**inputs)

    start_idx = torch.argmax(output.start_logits)
    end_idx = torch.argmax(output.end_logits)

    answer_ids = inputs["input_ids"][0, start_idx:end_idx+1]

    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)


    return answer

if __name__ == "__main__":
    jd_text = """
        We are looking for an AI Engineer with experience in
        Python, Machine Learning, Docker, FastAPI and MLOps."""
    
    question = "What skills are required?"

    answer = answer_question(jd_text, question)

    print(answer)

