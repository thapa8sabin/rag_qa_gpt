# 🎯 Quiz-GPT

**Quiz-GPT** is an end-to-end system for generating, evaluating, storing, and serving high-quality educational quizzes from a knowledge base.

It combines semantic retrieval, transformer-based question generation, a fine-tunable quality evaluator, and a natural-language chat interface — all running locally.

---

## ✨ Features

| Feature                           | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| 🔍 **Semantic Retrieval**         | Finds relevant context from a Qdrant vector database using embeddings    |
| 🧠 **Question Generation**        | Creates reading-comprehension style Q-A pairs with a T5 model            |
| 🏆 **Quality Ranking**            | Filters low-quality pairs using a fine-tuned QA Evaluator                |
| 🎓 **Custom Model Fine-Tuning**   | Train your own QA Evaluator on golden datasets + public data (see below) |
| 💾 **Quiz Persistence**           | Save, list, and retrieve quizzes with automatic deduplication            |
| 💬 **Natural Language Interface** | Chat-driven control via Gradio + Strands Agent + Ollama                  |
| 🏠 **Fully Local Stack**          | Ollama + Qdrant + FastMCP — no external API keys required                |

### 🎓 Highlight: Fine-Tuning the QA Evaluator

One of the most powerful capabilities of Quiz-GPT is the ability to **fine-tune the quality ranking model** on your own domain data.

- Supply a **golden dataset** of high-quality question–answer pairs (with context).
- Optionally mix in public Hugging Face QA datasets for scale.
- The built-in `Trainer` performs supervised fine-tuning of a BERT-style sequence classifier.
- Positive pairs (good Q-A) and automatically generated negative pairs (corrupted answers) teach the model to distinguish quality.
- The best checkpoint (by validation accuracy) is saved and can be plugged straight back into the generation pipeline.

This lets you adapt the system to specialized domains (civil service exams, medical quizzes, local language content, etc.).

→ Full details: [docs/training.md](docs/training.md)

---

## 🏗️ High-Level Architecture

```
User (Gradio Chat)
        │
        ▼
┌───────────────────┐
│  Strands Agent    │  ← Ollama (qwen3.5:2b)
│  + Tool wrappers  │
└─────────┬─────────┘
          │ StdioTransport
          ▼
┌───────────────────┐
│  FastMCP Server   │  ("Quiz-GPT")
│  (server.py)      │
└─────────┬─────────┘
          │
    ┌─────┴─────────────┬──────────────────────────┐
    ▼                   ▼                          ▼
Qdrant KB       Training + Evaluation        Quiz Store
(embeddings)      (fine tuning)          (SQLite/DB)
```

**Flow**

1. User asks for a quiz on a topic.
2. Agent calls `generate_questions_and_answers_pairs` (or `generate_and_save_quiz`).
3. Server retrieves relevant context → generates candidate pairs with T5 → ranks them with the Evaluator.
4. Optionally validates, deduplicates, and stores the quiz.

---

## 📚 Documentation Index

| Document                                                   | Description                                         |
| ---------------------------------------------------------- | --------------------------------------------------- |
| [docs/setup.md](docs/setup.md)                             | ⚙️ Installation, dependencies, Ollama, Qdrant       |
| [docs/knowledge-base.md](docs/knowledge-base.md)           | 📥 Ingesting documents into the vector store        |
| [docs/mcp-server.md](docs/mcp-server.md)                   | 🛠️ MCP server tools and helper functions            |
| [docs/client.md](docs/client.md)                           | 💬 Gradio client, agent behaviour, tools            |
| [docs/question-generation.md](docs/question-generation.md) | 🧠 QuestionGenerator & QAEvaluator pipelines        |
| [docs/training.md](docs/training.md)                       | 🎓 Fine-tuning the custom QA Evaluator              |
| [docs/feasibility.md](docs/feasibility.md)                 | 🔎 Feasibility, limitations & real-world usefulness |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.14.3
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for local Qdrant) **or** a Qdrant Cloud account
- Ollama

### 2. Install

```bash
# Install uv (Linux / macOS)
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh

# Enter project
cd quiz-gpt

# Install dependencies
uv sync
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:2b
QDRANT_URL=http://localhost:6333
QUIZ_SERVER_DIR=/absolute/path/to/quiz-gpt
```

### 4. Start supporting services

```bash
# Ollama
ollama pull qwen3.5:2b

# Qdrant (local)
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Ingest knowledge base

Place `.md` / `.txt` files in the docs folder, then:

```bash
uv run python -m src.injest
```

### 6. Train the QA Evaluator

```bash
uv run python -m src.training.qa_eval_train --golden_path golden/golden_dataset.json --use_public --public_sample_size 5000 --golden_upsample 20 --device cpu --epochs 3 --train_batch_size 8 --valid_batch_size 16 --learning_rate 2e-5 --valid_ratio 0.05
```

### 7. Run the application

```bash
# Quiz-GPT MCP Server
uv run fastmcp run server

# Gradio UI (launches MCP server on demand)
uv run python -m client
```

Open **http://127.0.0.1:7860**

---

## 🧩 Models Used

| Component               | Model / Checkpoint                       | Role                                   |
| ----------------------- | ---------------------------------------- | -------------------------------------- |
| 🧠 Question Generator   | `iarfmoose/t5-base-question-generator`   | Generates candidate Q-A pairs          |
| 🏆 Base QA Evaluator    | `iarfmoose/bert-base-cased-qa-evaluator` | Starting point for quality scoring     |
| 🎓 Fine-tuned Evaluator | `qa-evaluator-finetuned` (local)         | Domain-adapted quality ranking         |
| 💬 Chat LLM             | `qwen3.5:2b` (via Ollama)                | Tool-calling agent in the UI           |
| 🔍 Embeddings           | `all-MiniLM-L6-v2`                       | Document & query embeddings for Qdrant |

---

## 📁 Project Structure (simplified)

```
quiz-gpt/
├── client.py                   # Gradio + Strands agent
├── server.py                   # FastMCP server
├── data/                       # Knowedge base of qa pairs
├── datasets/                   # saved custom model from training
│   ├── qa-evaluator-finetuned/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── model.tokenizer_config.json
│   │   └── tokenizer.json
├── docs/                       # Documentation of project
├── src/
│   ├── generation/             # QuestionGenerator, QAEvaluator
│   ├── retrieval/              # Retrieved vector points from Qdrant
│   ├── storage/                # Quiz store + deduplication
│   ├── training/               # Trainer + qa_eval_train script
│   └── injest.py               # Knowledge-base ingestion
├── golden/                     # Golden dataset for training
├── quiz.db                     # SQLite db for storing generated qa pairs
└── .env
```

---

## 📄 Licence & Acknowledgements

- Question generation pipeline adapted from [AMontgomerie/question_generator](https://github.com/AMontgomerie/question_generator)
- Base models hosted on Hugging Face by [iarfmoose](https://huggingface.co/iarfmoose)

---

For deeper technical explanations, start with the **Documentation Index** above.  
For real-world applicability, limitations, and future directions (including Nepali / Lok Sewa use-cases), see [docs/feasibility.md](docs/feasibility.md).
