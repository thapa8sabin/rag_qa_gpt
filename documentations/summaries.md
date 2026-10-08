## Introduction

Quiz-GPT system, responsible for generating quiz quiz questions from educational articles using LLMs. It coordinates semantic retrieval, reranking, and question generation through an embedding-based pipeline.

It uses two transformer models for generating questions an answers pairs using queried topic(s):

1. **Question Generator**: creates candidate question–answer pairs from a given text.
2. **QA Evaluator**: scores those pairs so only the highest-quality ones are kept.

## Purpose

1. Converts query text into a numeric vector representation using the CrossEncoder model.
   - Takes input `query` string
   - Encodes it through `CrossEncoder/ms-marco-MiniLM-L-6-v2`
   - Returns list of float vectors representing semantic meaning
2. Finds most relevant documents from a corpus using semantic similarity to user queries.
   - Converts each `query` into an embedding vector
   - Searches the document index for matches above a threshold score
   - Sorts and selects `top_k` highest-relevant results
3. Performs high-precision ranking on retrieved documents using `CrossEncoder`.
   - Constructs query-content pairs from `query_embedding` and document text
   - Passes them to `CrossEncoder` for contextualized scoring
   - Sorts results by score (descending order)
   - Returns `top-k` ranked results with context scores
4. Generates quiz questions from a summarized context using the LLM.
   - Joins and summarizes retrieved docs into context string
   - Passes to `QuestionGenerator` as an article
   - Returns 5 generated questions in list of dicts
   - Uses prompt templates (`default: quiz format`)

This pipeline ensures high-quality quizzes by using both semantic understanding and context-based retrieval rather than keyword matching.

## Project Setup

### Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# ====================MAC OS and Linux============================
# We are using uv version 0.12.19 for this project
# Request a specific version by including it in the URL:
#
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh


# ====================Windows============================
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.23/install.ps1 | iex"

```

### Install required dependencies for the project

```bash
    uv sync
```

These dependies will be added:

```text
    "fastmcp>=4.0.11",
    "langchain-text-splitters>=1.1.3",
    "qdrant-client>=1.19.1",
    "sentencepiece>=0.2.0",
    "sentence-transformers>=6.1.0",
    "tiktoken>=0.14.0",
    "torch>=2.14.1",
    "transformers>=5.19.0",
    "python-dotenv>=1.2.4",
    "accelerate>=1.15.0",
    "gradio>=5.0",
    "strands-agents[ollama,openai]>=1.0.0",
    "numpy>=2.5.3",
    "datasets>=5.1.0",
    "scikit-learn>=1.9.1",
    "tqdm>=4.70.1",
    "pandas>=3.0.6",
```

### Setup Ollama

```bash
# ----------------- Mac and Linux -------------
curl -fsSL https://ollama.com/install.sh | sh

# ----------------- Windows ----------------------
irm https://ollama.com/install.ps1 | iex
```

Run the Model using Ollama

```bash
# Pull the qwen model for our chat
ollama pull qwen3.5:2b # we are using this model for quiz interface

# this will download the qwen3.5:2b model and will be accessible using ollama mcps url: http://localhost:11434/api/tags [default]. If model is still not found try launching model

# Launch the model (Optional)
ollama launch qwen3.5:2b
```

### Setup Qdrant Db

```bash
# ----------- Local Setup -------------------
# Prerequisites: Docker
# Download the docker from https://www.docker.com/ then
docker pull qdrant/qdrant
# Then add QDRANT_URL=http://localhost:6333 in .env file.

#---------- Qdrant Cloud --------------------
# Login to https://qdrant.tech/documentation/web-ui/
# Access console from https://qdrant.tech/documentation/web-ui/ and generate token and add it to .env file as:
# QDRANT_URL=https://...
```

### Injest in Vector DB

This Python script automates the ingestion of unstructured text documents into a Qdrant vector database to facilitate efficient semantic search and retrieval operations. It first verifies that an input docs directory exists, then iterates through all Markdown and plain text files within it to process them individually. For each file, the code reads the content, splits it into smaller overlapping chunks using a RecursiveCharacterTextSplitter for better indexing efficiency, encodes those chunks into 384-dimensional embeddings using a pre-trained SentenceTransformer model (all-MiniLM-L6-v2), and assigns metadata to track the original source document. The script then creates a new collection named 'quiz_knowledge_base' with cosine distance metrics if it does not already exist, upserts these embedded vectors into Qdrant for storage, and confirms successful completion by printing a success message once all documents are uploaded.

Run this command in terminal to injest knowledges in vector db.

```bash
# make sure there are files in folder with either *.md or *.txt formats and run
uv run python -m src.injest
```

## Quiz-GPT MCP Server

This is a FastMCP server named “Quiz-GPT” that exposes tools for generating, storing, and retrieving quizzes backed by a knowledge base and a quiz database.
Main pieces

Loads environment variables and sets up a FastMCP instance.
Imports question generation (generate_questions), quiz storage (quiz_store), and deduplication logic.

### Helper functions

\_validate_pairs – Ensures every question/answer pair has non-empty content.
\_store – Validates, deduplicates (within the batch + against the DB), then saves the quiz. Returns a human-readable status message about what was saved or skipped.

### Exposed MCP tools

- `search_knowledge_base`: Takes a query, generates relevant Q&A pairs from the knowledge base, and returns them as text.
- `save_quiz`: Saves a list of question/answer pairs under a given topic (with validation + deduplication).
- `generate_and_save_quiz`: Combines the above: generates Q&A pairs for a query and immediately stores them as a quiz.
- `list_quizzes`: Returns a summary of all stored quizzes (id, topic, question count, creation time).
- `get_quiz`: Fetches a quiz by ID; optionally includes or hides the answers.

Run the server using this command

```bash
# Initialize 'Quiz-GPT' MCP server using Standard I/O transport.
uv run fastmcp run server

# The chat uses Strands to call the `search_knowledge_base` FastMCP tool and displays the returned questions and answers.
```

## Quiz-GPT Client

This is the client-side Gradio chat app for Quiz-GPT. It connects a local Ollama LLM (via the Strands Agent framework) to the MCP quiz server so users can generate, save, list, and retrieve quizzes through natural language.

### Setup

Loads environment variables (Ollama host/model, path to the quiz server).
Uses StdioTransport to launch the MCP server (uv run server.py) on demand for each tool call.
Wraps every server tool in a synchronous @tool function that runs the async MCP call via asyncio.run.

Run this command to start the application:

```bash
uv run python -m client
# Local URL:  http://127.0.0.1:7860
```

### Tools exposed to the agent

- `search_knowledge_base`: Generate Q&A pairs (does not save).
- `save_quiz`: Store a list of question/answer pairs under a topic.
- `generate_and_save_quiz`: Generate + save in one step.
- `list_quizzes`: Show all stored quizzes.
- `get_quiz`: Fetch a quiz by ID (optionally hide answers).

### Agent behavior

System prompt instructs the model when to call each tool, to present results faithfully, collect at least 10 questions, and turn them into quiz quizzes.
Temperature is set to 0 so the model doesn’t reword questions/answers.

### UI

Simple Gradio ChatInterface titled “Quiz-GPT”.
Handles empty input and Ollama connection errors gracefully.
Includes example prompts for common workflows.

## Training a model

Based on the knowledge base available in `docs` folder train a new qa generator model along with hugging face public library

This script fine-tunes a Sequence Classification model (such as BERT) to evaluate question-answering pairs using custom dataset handling and PyTorch/Hugging Face utilities. It loads, normalizes, and flattens a custom local "golden" dataset (from JSON, JSONL, or CSV formats) and optionally merges it with a subsampled public QA evaluation dataset (iarfmoose/qa_evaluator). After upsampling the domain-specific golden data to balance the mixture, the script splits the combined dataset into training and validation sets, tokenizes the input text, and trains the classification model over a specified number of epochs before saving the fine-tuned weights for downstream evaluation.

|                        |                                        |                                                                                                            |
| :--------------------- | :------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **QA Generator Model** | iarfmoose/t5-base-question-generator   | Responsible for generating questions and answers paris                                                     |
| **QA Evaluator Model** | iarfmoose/bert-base-cased-qa-evaluator | Responsible for evaluating those generated qa pairs                                                        |
| **QA Train Model**     | qa-evaluator-finetuned                 | Trained model using golden datasets and public qa datasets from hugging face to generate relevent qa pairs |
| **Public Dataset**     | iarfmoose/qa_evaluato                  | Public dataset to evaluate qa pairs from huggingface                                                       |

Run the following command to train the model.

```bash
# this will run the script in efficiently to train a model in cpu.
# use cuda for better improvements as feasiblity study later
 uv run python -m src.training.qa_eval_train --golden_path golden/golden_dataset.json --use_public --public_sample_size 5000 --golden_upsample 20 --device cpu --epochs 3 --train_batch_size 8 --valid_batch_size 16 --learning_rate 2e-5 --valid_ratio 0.05
```

What the new flags do
| Flag | Meaning | Suggested Value |
| :--------------------- | :------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
|`--use_public` | Load the public HuggingFace dataset | always on |
| `--public_sample_size` | How many public examples to keep (CPU) | 3000–8000|
| `--golden_upsample` | Repeat each of your golden examples this many times | 10–30|
|`--valid_ratio` | Fraction held out for validation | 0.05 (so you still get a decent validation set) |
With the numbers above you get roughly:

20 golden × 20 upsample = 400 domain examples

5000 public examples

→ ~5400 total → train ≈ 5100 / valid ≈ 270

That is already enough for the model to learn something useful while still finishing in a reasonable time on CPU.

### How the evaluation / training works ?

1. You supply a public dataset from huggingface along with the golden list of real (question, answer) pairs.
2. The dataset creates:
   - label = 1 → original golden pair
   - label = 0 → corrupted pair (answer swapped with another golden answer, or a fragment of the question is used as the answer)
3. A BERT sequence-classification head is trained to distinguish good vs bad pairs.
4. The best checkpoint (by validation accuracy) is saved.

## Trainer

The trainer takes a pre-trained model + tokenizer, a training dataset, and a validation dataset, then runs standard supervised fine-tuning with early-best-model saving based on validation performance.

### Data sources used for training

1. Golden dataset (the JSON you showed)
   - Hand-crafted high-quality examples organized by topic (arts_and_literature, general_knowledge, geography, science_and_nature, sports_and_pop_culture, …).
   - Each item contains:
     - `question`
     - `context` (one or more supporting passages)
     - `ideal_answer`

   - These serve as reliable **positive** examples (good question–answer pairs that are answerable from the given context).

2. Public Hugging Face datasets
   - Additional large-scale QA / reading-comprehension datasets (e.g. SQuAD-style or similar) are loaded and mixed in.
   - They provide volume and diversity that the relatively small golden set cannot supply alone.

### How the data is turned into training examples

Typical construction for a QA-evaluator (binary or quality classifier):

- Positive pairs: (question, ideal_answer) or (question, context + answer) taken from the golden set and the public datasets → label = good / high quality.
- Negative pairs: created by mismatching questions and answers, using incorrect answers, or pairing a question with unrelated context → label = bad / low quality.

This produces a supervised classification dataset that the model learns from.

### What the Trainer does with that data

Once the `train_set` and `valid_set` (PyTorch Datasets built from the golden + HF data) are ready, the Trainer:

- Fine-tunes a sequence-classification model (usually a BERT-style model) with AdamW.
- Runs a simple epoch loop with gradient clipping.
- Evaluates on the validation split using either accuracy (default) or loss.
- Saves the best checkpoint (model + tokenizer) whenever validation performance improves.

### Result

The fine-tuned model becomes the QAEvaluator used later by `Question Generator`. It scores newly generated question–answer pairs so that only the highest-quality ones are kept for the quiz.
In short: the golden dataset supplies clean, domain-relevant positive examples; public HF datasets add scale; the Trainer then fine-tunes a classifier on the resulting positive/negative pairs.

## Question Generator and Evaluation

- Loads a fine-tuned T5 model (`iarfmoose/t5-base-question-generator`).
- Moves it to GPU if available.
- Instantiates a QAEvaluator (using a local fine-tuned checkpoint by default).

### QA Generator pipeline

1. Prepare inputs
   - Split the article into manageable segments (≤ ~490 tokens) so they fit the model.
   - Inside each segment, split the text into sentences (and further split very long sentences).
   - For every sentence treat it as a potential answer.
   - Build a model input of the form:
     `<answer> {sentence} <context> {full segment text}`

2. Generate questions
   - Feed each prepared input into the T5 model.
   - The model produces a natural-language question whose answer is expected to be that sentence.
   - Decode the output and collect the questions.

3. Quality filtering
   - The generated questions + their answers are passed to QAEvaluator.
   - Scores are obtained and the pairs are ranked.
   - Only the top num_questions (default 10) are returned.

### QA Evaluation pipeline

1. Encode qa pairs
   - For each (question, answer) pair, tokenize them together as a text-pair (question as text, answer as text_pair).
   - Pad/truncate to 512 tokens.

2. Score
   - Run the classifier on each encoded pair.
   - Extract the logit for the positive class.
   - Higher score = higher predicted quality.

3. Ranking
   - Return the indices of the pairs sorted by descending score.
   - `Question Generator` uses these indices to keep only the best pairs.

In short: the generator proposes many candidate Q–A pairs by treating sentences as answers and asking T5 to invent questions; the evaluator ranks them with a classifier so only the most coherent / answerable ones are kept.

### Custom Evaluator

After running `qa_eval_train.py`, you will have a directory (e.g. `./qa-evaluator-finetuned`) containing:

- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- tokenizer files

**After Training** (optional) add your existing `QA Evaluator` to the new folder

```python
# go to file src/generation/questionsgenerator.py:236
QAE_PRETRAINED = "/path/to/qa-evaluator-finetuned"
```

**Note**: This is based on research and still need more improvements the links to implementation is provided below

- [Inference Providers](https://huggingface.co/iarfmoose/t5-base-question-generator)

- [github project](https://github.com/AMontgomerie/question_generator/tree/master)
