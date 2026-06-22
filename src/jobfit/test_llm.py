from .llm import generate_text

prompt = """
Improve this resume bullet for a Junior ML Engineer role:

Built machine learning models using Python.

Be specific, add metrics if possible, and use strong action verbs.
"""

for temp in [0.2, 0.7, 1.2]:
    print("=" * 50)
    print(f"Temperature: {temp}")
    result = generate_text(prompt, temperature=temp)
    print(result)
    print()