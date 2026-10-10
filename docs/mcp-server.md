# MCP Server (Quiz-GPT)

The server is implemented with **FastMCP** and exposes a set of tools that the client agent can call.

## Overview

File: `server.py`

- Loads environment variables via `python-dotenv`.
- Instantiates `FastMCP("Quiz-GPT")`.
- Imports:
  - `generate_questions` (question generation pipeline)
  - `quiz_store` (persistence layer)
  - `dedupe_pairs` (deduplication logic)

## Helper functions

### `_validate_pairs(questions)`

- Ensures every item contains non-empty `question` and `answer` fields.
- Raises `ValueError` on the first invalid item.
- Returns a cleaned list of dictionaries.

### `_store(topic, questions)`

1. Validates the pairs.
2. Deduplicates both within the current batch and against the existing database.
3. Saves the remaining unique pairs under the given topic.
4. Returns a human-readable status string describing how many questions were saved and how many were skipped.

## Exposed tools

| Tool                                   | Description                                                               | Side effects     |
| -------------------------------------- | ------------------------------------------------------------------------- | ---------------- |
| `generate_questions_and_answers_pairs` | Retrieves relevant context and generates Q-A pairs for a query.           | None (read-only) |
| `save_quiz`                            | Stores a list of question–answer pairs under a topic.                     | Writes to DB     |
| `generate_and_save_quiz`               | Generates pairs for a query **and** stores them in one step.              | Writes to DB     |
| `list_quizzes`                         | Returns id, topic, question count and creation time of every stored quiz. | None             |
| `get_quiz`                             | Fetches a quiz by id. Optional flag to hide answers.                      | None             |

### Tool signatures (simplified)

```python
generate_questions(query: str) -> str
save_quiz(topic: str, questions: list[dict[str, str]]) -> str
generate_and_save_quiz(query: str) -> str
list_quizzes() -> str
get_quiz(quiz_id: int, include_answers: bool = True) -> str
```

## Starting the server

```bash
uv run fastmcp run server
# or
uv run python server.py
```

The server listens on standard input/output (Stdio transport), which is how the Gradio client communicates with it.
