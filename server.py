from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

from src.generation.question_llm import generate_questions
from src.storage import quiz_store
from src.storage.dedupe import dedupe_pairs

mcp = FastMCP("Quiz-GPT")


def _store(topic: str, questions: list[dict[str, str]]) -> str:
    """Validate, dedupe (within the batch and against the DB) and save a quiz."""
    pairs = _validate_pairs(questions)
    unique_pairs, dropped = dedupe_pairs(pairs)
    if not unique_pairs:
        return f"Nothing saved: all {len(pairs)} questions were duplicates or low quality."

    result = quiz_store.save_quiz(topic, unique_pairs)
    skipped = dropped + result["skipped"]
    if result["quiz_id"] is None:
        return f"Nothing saved: all {len(pairs)} questions already exist in the database."
    return (
        f"Saved quiz {result['quiz_id']} ('{topic}') with {result['saved']} questions. "
        f"Skipped {skipped} duplicate/low-quality questions."
    )


def _validate_pairs(questions: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = []
    for i, qa in enumerate(questions, start=1):
        question = (qa.get("question") or "").strip()
        answer = (qa.get("answer") or "").strip()
        if not question or not answer:
            raise ValueError(f"Item {i} needs non-empty 'question' and 'answer'.")
        cleaned.append({"question": question, "answer": answer})
    return cleaned


@mcp.tool
def generate_questions_and_answers_pairs(query: str) -> str:
    """ Generates a paired set consisting of a question and its corresponding correct answer based on a source text, knowledge graph, or dataset."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    question_answer_pairs = generate_questions(query)

    if not question_answer_pairs:
        return "No relevant context found for the given query."

    results = []
    for qa_pair in question_answer_pairs:
        question = qa_pair.get("question", "No question found.")
        answer = qa_pair.get("answer", "No answer found.")
        results.append(f"Question: {question}\nAnswer: {answer}")

    return "\n\n".join(results)


@mcp.tool
def save_quiz(topic: str, questions: list[dict[str, str]]) -> str:
    """Store a quiz. `questions` is a list of {"question": ..., "answer": ...} objects."""
    if not topic.strip():
        raise ValueError("Topic must not be empty.")
    if not questions:
        raise ValueError("At least one question is required.")

    return _store(topic.strip(), questions)


@mcp.tool
def generate_and_save_quiz(query: str) -> str:
    """Generate questions from the knowledge base for a topic and store them as a quiz."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    qa_pairs = generate_questions(query)
    if not qa_pairs:
        return "No relevant context found for the given query. Nothing saved."

    return _store(query.strip(), qa_pairs)


@mcp.tool
def list_quizzes() -> str:
    """List all stored quizzes with their id, topic, creation time and question count."""
    quizzes = quiz_store.list_quizzes()
    if not quizzes:
        return "No quizzes stored yet."

    return "\n".join(
        f"[{q['id']}] {q['topic']} - {q['question_count']} questions (created {q['created_at']})"
        for q in quizzes
    )


@mcp.tool
def get_quiz(quiz_id: int, include_answers: bool = True) -> str:
    """Fetch a stored quiz by id. Set include_answers=False to get questions only."""
    quiz = quiz_store.get_quiz(quiz_id)
    if quiz is None:
        return f"No quiz found with id {quiz_id}."

    lines = [f"Quiz {quiz['id']}: {quiz['topic']}"]
    for q in quiz["questions"]:
        lines.append(f"\n{q['position']}) Question: {q['question']}")
        if include_answers:
            lines.append(f"   Answer: {q['answer']}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()