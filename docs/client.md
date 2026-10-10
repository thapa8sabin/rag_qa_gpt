# Client (Gradio + Strands Agent)

The client provides a conversational interface to the Quiz-GPT MCP server.

## Overview

File: `client.py`

- Launches a Gradio `ChatInterface`.
- Uses the **Strands Agents** framework with an Ollama-backed model.
- For every tool call the client starts a short-lived MCP server process via `StdioTransport` and `uv run server.py`.

## Configuration

Environment variables (loaded from the project-root `.env`):

| Variable          | Default                  | Purpose                           |
| ----------------- | ------------------------ | --------------------------------- |
| `OLLAMA_HOST`     | `http://localhost:11434` | Ollama endpoint                   |
| `OLLAMA_MODEL`    | `qwen3.5:2b`             | Model used by the agent           |
| `QUIZ_SERVER_DIR` | (must be set)            | Absolute path to the project root |

## Tool wrappers

Each MCP tool is exposed to the agent as a synchronous `@tool` function that internally runs the async call with `asyncio.run`:

- `generate_questions(query)`
- `save_quiz(topic, questions)`
- `generate_and_save_quiz(query)`
- `list_quizzes()`
- `get_quiz(quiz_id, include_answers=True)`

## Agent behaviour (system prompt)

The system prompt instructs the model to:

- Call `generate_questions` exactly once when a quiz is requested (do **not** save automatically).
- Call `generate_and_save_quiz` only when the user explicitly asks to generate **and** save.
- When saving an already-displayed quiz, pass the exact question/answer pairs that were shown.
- Always report the server’s result (how many questions were saved / skipped).
- Use `list_quizzes` and `get_quiz` for listing and retrieving stored quizzes.
- Collect at least 10 questions and present them as a quiz (including multiple-choice options when appropriate).
- Present all tool results faithfully in Markdown.

Temperature is set to `0` so the model does not reword the generated questions or answers.

## Starting the UI

```bash
uv run python -m client
```

The Gradio interface is served at http://127.0.0.1:7860.

Example prompts provided in the UI:

- “General Knowledge”
- “Generate and save a quiz on Geography”
- “List my saved quizzes”
- “Show quiz 1 without answers”
