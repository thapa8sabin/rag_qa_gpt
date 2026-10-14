import asyncio
import os
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from strands import Agent, tool
from strands.models.ollama import OllamaModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")

SERVER_DIR = os.getenv("QUIZ_SERVER_DIR", "C:\\Users\\User\\Desktop\\workspace\\quiz-gpt")

SYSTEM_PROMPT = """
You are Quiz-GPT. You generate, store, list and fetch quizzes using your tools.

Generating a quiz:
- When the user asks for a quiz on a topic, call generate_questions_and_answers_pairs exactly once.
- Show the returned questions and answers in a clear.
- Do NOT save the quiz unless the user asks you to.

Saving a quiz:
- If the user asks to generate AND save a quiz in one go, call generate_and_save_quiz once.
- If the user asks to save a quiz that was already shown, call save_quiz with the topic and
  the exact question/answer pairs you displayed. Never alter their wording.
- The server skips duplicate and low-quality questions. Always tell the user the tool's
  result, including how many questions were saved and skipped.

Listing and fetching:
- To show stored quizzes, call list_quizzes.
- To open a stored quiz, call get_quiz with its id. If the user wants to take the quiz or
  asks for questions only, set include_answers to false.
- If the user refers to a quiz by topic and you don't know its id, call list_quizzes first.

Present tool results faithfully in Markdown. If a tool says nothing was found or saved,
say so plainly.
- Collect atleast 10 qustions.
- Show only questions and display as a quiz.
"""


async def _call_server_tool(name: str, arguments: dict[str, Any]) -> str:
    """Start the quiz MCP server, call one tool, and return its text output."""
    transport = StdioTransport(
        command="uv",
        args=["--directory", SERVER_DIR, "run", "server.py"],
    )
    async with Client(transport) as client:
        result = await client.call_tool(name, arguments)

    text_blocks = [block.text for block in result.content if hasattr(block, "text")]
    if not text_blocks:
        raise RuntimeError(f"The quiz server returned no content for '{name}'.")
    return "\n\n".join(text_blocks)


@tool
def generate_questions_and_answers_pairs(query: str) -> str:
    """Generate quiz questions and answers for a topic from the knowledge base.
    Does not save anything."""
    if not query.strip():
        raise ValueError("Please enter a topic for the quiz.")
    return asyncio.run(_call_server_tool("generate_questions_and_answers_pairs", {"query": query}))


@tool
def save_quiz(topic: str, questions: list[dict[str, str]]) -> str:
    """Store a quiz in the database.

    Args:
        topic: Short title/topic of the quiz.
        questions: List of objects like {"question": "...", "answer": "..."}.
            Duplicates and low-quality pairs are skipped by the server.
    """
    if not topic.strip():
        raise ValueError("A topic is required to save a quiz.")
    if not questions:
        raise ValueError("There are no questions to save.")
    return asyncio.run(
        _call_server_tool("save_quiz", {"topic": topic, "questions": questions})
    )


@tool
def generate_and_save_quiz(query: str) -> str:
    """Generate quiz questions for a topic from the knowledge base and store them
    in one step. Duplicates and low-quality pairs are skipped."""
    if not query.strip():
        raise ValueError("Please enter a topic for the quiz.")
    return asyncio.run(_call_server_tool("generate_and_save_quiz", {"query": query}))


@tool
def list_quizzes() -> str:
    """List all stored quizzes with their id, topic, creation time and question count."""
    return asyncio.run(_call_server_tool("list_quizzes", {}))


@tool
def get_quiz(quiz_id: int, include_answers: bool = True) -> str:
    """Fetch a stored quiz by its id.

    Args:
        quiz_id: The id shown by list_quizzes.
        include_answers: Set to False to return questions only (for taking the quiz).
    """
    return asyncio.run(
        _call_server_tool(
            "get_quiz", {"quiz_id": quiz_id, "include_answers": include_answers}
        )
    )


ollama_model = OllamaModel(
    host=OLLAMA_HOST,
    model_id=OLLAMA_MODEL,  # must be a model that supports tool calling
    temperature=0.4,  # keeps the model from rewording questions/answers
)

agent = Agent(
    model=ollama_model,
    tools=[
        generate_questions_and_answers_pairs,
        save_quiz,
        generate_and_save_quiz,
        list_quizzes,
        get_quiz,
    ],
    system_prompt=SYSTEM_PROMPT,
)


def chat(message: str, history: list[dict[str, str]]) -> str:
    del history
    if not message.strip():
        raise gr.Error("Enter a request, e.g. a quiz topic.")

    try:
        return str(agent(message))
    except Exception as exc:
        if "connect" in str(exc).lower():
            raise gr.Error(
                f"Could not reach Ollama at {OLLAMA_HOST}. Is it running "
                f"(`ollama serve`) and is '{OLLAMA_MODEL}' pulled?"
            ) from exc
        raise


demo = gr.ChatInterface(
    fn=chat,
    title="Quiz-GPT",
    description=(
        "Generate a quiz from your knowledge base, then save it, list saved quizzes, "
        "or open one by id."
    ),
    examples=[
        "General Knowledge",
        "Generate and save a quiz on Geography",
        "Science and Nature",
        "Save this quiz",
        "List my saved quizzes",
        "Show quiz 1 without answers",
    ],
)


if __name__ == "__main__":
    demo.launch()