# Knowledge Base Ingestion

Quiz-GPT relies on a **Qdrant** vector database that stores embeddings of educational documents. Semantic search over this collection supplies the context used by the question generator.

## Purpose of the ingestion pipeline

The ingestion script (`src/injest.py`) performs the following steps for every document:

1. Verifies that the source directory exists.
2. Iterates over all `.md` and `.txt` files.
3. Reads the full content of each file.
4. Splits the text into overlapping chunks with `RecursiveCharacterTextSplitter` (LangChain).
5. Encodes each chunk into a 384-dimensional vector using the SentenceTransformer model `all-MiniLM-L6-v2`.
6. Attaches metadata (source filename, chunk index, etc.).
7. Creates the collection `quiz_knowledge_base` (cosine distance) if it does not already exist.
8. Upserts the vectors into Qdrant.

## Running ingestion

Place your educational material (Markdown or plain-text files) into the designated docs folder, then execute:

```bash
uv run python -m src.injest
```

A success message is printed once every document has been processed and uploaded.

## Design notes

- Chunking with overlap improves retrieval of relevant context even when a topic spans paragraph boundaries.
- Cosine similarity is used because the embedding model was trained with cosine distance.
- The collection name `quiz_knowledge_base` is fixed; change it only if you also update the retrieval code.

After ingestion is complete the knowledge base is ready for use by the MCP tools `search_knowledge_base` and `generate_and_save_quiz`.
