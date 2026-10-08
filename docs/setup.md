# Setup & Installation

This guide covers everything required to run Quiz-GPT locally.

## 1. Install uv

We use the [uv](https://docs.astral.sh/uv/) package manager (pinned version recommended).

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.23/install.ps1 | iex"
```

## 2. Install project dependencies

From the project root:

```bash
uv sync
```

Key packages that will be installed:

| Package                        | Purpose                                      |
|--------------------------------|----------------------------------------------|
| `fastmcp`                      | MCP server framework                         |
| `strands-agents[ollama]`       | Agent framework + Ollama integration         |
| `gradio`                       | Chat UI                                      |
| `transformers` / `torch`       | Question generation & evaluation models      |
| `sentence-transformers`        | Embedding model for Qdrant                   |
| `qdrant-client`                | Vector database client                       |
| `datasets` / `scikit-learn`    | Training the QA Evaluator                    |
| `python-dotenv`                | Environment variable loading                 |

## 3. Install & configure Ollama

**Linux / macOS**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

```powershell
irm https://ollama.com/install.ps1 | iex
```

Pull the model used by the chat interface:

```bash
ollama pull qwen3.5:2b
```

The model will be available at the default endpoint `http://localhost:11434`.

## 4. Set up Qdrant

### Option A – Local (Docker)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

Then set in `.env`:

```env
QDRANT_URL=http://localhost:6333
```

### Option B – Qdrant Cloud

1. Create an account at [qdrant.tech](https://qdrant.tech).
2. Create a cluster and copy the URL + API key.
3. Add to `.env`:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

## 5. Environment variables

Create a `.env` file in the project root:

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:2b

# Qdrant
QDRANT_URL=http://localhost:6333

# Path used by the client to launch the MCP server
QUIZ_SERVER_DIR=/absolute/path/to/quiz-gpt
```

## 6. Verify installation

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check Qdrant
curl http://localhost:6333/collections
```

You are now ready to ingest documents and start the application.
