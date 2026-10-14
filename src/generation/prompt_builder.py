def generate_prompt(qa_pairs: dict[str, str]) -> str:
    """Build one multiple-choice generation prompt for each question-and-answer pair."""
    prompt_instruction = """
You are an expert educational assessment creator. Convert the input question-and-answer pairs into one multiple-choice questions.

Requirements:
- Provide exactly four options labeled A, B, C, and D.
- Exactly one option must be correct; the other three must be plausible, relevant distractors.
- Vary the correct answer's position rather than always placing it under A.
- Preserve the factual meaning of the supplied answer.
- Return only valid JSON matching this shape:
  {{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}
  The "answer" value must be the label of the correct option.
QA:
"""
    qa = "\n".join(f"Q: {qa_pair['question']}\nA:{qa_pair['answer']}" for qa_pair in qa_pairs)
    return prompt_instruction.join(qa)
