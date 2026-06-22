# Temperature Comparison — SmolLM2-1.7B-Instruct

**Model:** HuggingFaceTB/SmolLM2-1.7B-Instruct
**Prompt:** "Improve this resume bullet for a Junior ML Engineer role: Built machine learning models using Python. Be specific, add metrics if possible, and use strong action verbs."
**Run:** Local CPU (laptop)

---

## Temperature: 0.2 (low — deterministic, focused)

> "Developed and deployed high-performance machine learning models using Python, achieving a 95% accuracy rate in predictive modeling and a 20% reduction in processing time. Utilized TensorFlow and scikit-learn libraries to create scalable and efficient models, resulting in a 30% increase in model accuracy and a 15% reduction in training time."

**Observation:** Conservative, safe output. Sticks to factual improvements. Adds concrete numbers (95%, 20%, 30%). Low creativity.

---

## Temperature: 0.7 (medium — balanced)

> "Designed and developed a range of machine learning models using Python, with a focus on predictive analytics and natural language processing. Achieved a 95% accuracy rate in a supervised learning model, demonstrating a high level of performance and efficiency. Optimized model parameters to reduce training time by 30%, ensuring faster deployment and improved user experience."

**Observation:** More natural phrasing ("a range of", "with a focus on"). Introduces NLP as a domain. Still grounded but slightly more variety in structure.

---

## Temperature: 1.2 (high — creative, diverse)

> "Optimized and refined machine learning models using Python, leveraging advanced algorithms such as neural networks and supervised/unsupervised learning techniques. Accurately captured user input data, processed it, and generated high-quality predictions with exceptional precision. Achieved an 85% accuracy rate in testing. Collaborated with cross-functional teams to integrate models into production environments, and successfully deployed models that reduced error rates by 20%. Implemented continuous model evaluation and optimization, leading to improvements in system performance and efficiency."

**Observation:** Longest output. Adds entirely new dimensions — collaboration, production deployment, continuous evaluation. Mentions specific algorithm types. Slightly less coherent structure but most detailed.

---

## Summary

| Temp | Tone | Detail | Creativity |
|------|------|--------|------------|
| 0.2 | Safe, precise | Medium | Low |
| 0.7 | Natural, balanced | Medium | Medium |
| 1.2 | Expansive, varied | High | High |

**Recommendation for jobfit-rag-assistant:** Use **0.7** as default — balances coherence with useful variation.
