from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)




def summarize_jd(jd_text):

    inputs = tokenizer(jd_text,
                      return_tensors="pt",
                      truncation=True,
                      max_length=1024)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=20,
        max_length=80,
        do_sample=False
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

if __name__ == "__main__":

    jd_text = """
      We are looking for an AI Engineer with experience in
      Python, Machine Learning, Docker, FastAPI and MLOps.
      The candidate should have experience deploying models
      and building scalable ML systems."""
    
    print(summarize_jd(jd_text))