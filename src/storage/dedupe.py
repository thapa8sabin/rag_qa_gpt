import re

import numpy as np

from src.retrieval.query_embedder import embed_query

MIN_ANSWER_WORDS = 4
SIMILARITY_THRESHOLD = 0.9


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace so near-identical text matches."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text.lower())).strip()


def is_low_quality(qa: dict[str, str]) -> bool:
    """Drop pairs like 'Answer: 7.', all-caps headings, or answers that are just another question."""
    answer = qa["answer"].strip()
    if len(answer.split()) < MIN_ANSWER_WORDS:
        return True
    if answer.endswith("?"):
        return True
    letters = [c for c in answer if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    return False


def dedupe_pairs(
    qa_pairs: list[dict[str, str]],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[dict[str, str]], int]:
    """Remove low-quality pairs and duplicates from a batch.

    A pair is dropped if its question or answer matches an earlier one exactly
    (after normalisation) or if its question is semantically close to one
    already kept. Returns (kept_pairs, number_dropped).
    """
    kept, kept_vectors = [], []
    seen_questions, seen_answers = set(), set()

    for qa in qa_pairs:
        if is_low_quality(qa):
            continue

        q_key, a_key = normalize(qa["question"]), normalize(qa["answer"])
        if q_key in seen_questions or a_key in seen_answers:
            continue

        vector = np.array(embed_query(qa["question"]))
        vector = vector / (np.linalg.norm(vector) or 1.0)
        if any(float(vector @ v) >= similarity_threshold for v in kept_vectors):
            continue

        seen_questions.add(q_key)
        seen_answers.add(a_key)
        kept_vectors.append(vector)
        kept.append(qa)

    return kept, len(qa_pairs) - len(kept)