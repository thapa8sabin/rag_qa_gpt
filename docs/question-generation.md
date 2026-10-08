# Question Generation & Evaluation

This document describes the core NLP pipeline that turns retrieved context into ranked question–answer pairs.

## Components

| Class              | Base model                                      | Responsibility                              |
|--------------------|-------------------------------------------------|---------------------------------------------|
| `QuestionGenerator`| `iarfmoose/t5-base-question-generator` (T5)     | Propose candidate Q-A pairs                 |
| `QAEvaluator`      | Fine-tuned BERT (or `iarfmoose/bert-base-cased-qa-evaluator`) | Score and rank those pairs          |

## QuestionGenerator pipeline

### 1. Input preparation (`generate_qg_inputs`)

1. Split the article into segments of at most ~490 tokens (so they fit the model context).
2. Inside each segment, split the text into sentences (further splitting very long sentences).
3. Treat **every sentence** as a potential answer.
4. Build a model input of the form:

   ```
   <answer> {sentence} <context> {full segment text}
   ```

### 2. Question generation (`generate_questions_from_inputs`)

- Each prepared input is fed to the T5 model.
- The model generates a natural-language question whose expected answer is the chosen sentence.
- Generated questions are decoded and collected.

### 3. Quality filtering (optional but recommended)

- The generated questions and their answers are passed to `QAEvaluator`.
- Each pair receives a quality score.
- Pairs are ranked by descending score.
- Only the top `num_questions` (default 10) are retained.

If `use_evaluator=False`, every generated pair is returned without ranking.

### Output format

```python
[
  {"question": "What is ...?", "answer": "..."},
  ...
]
```

## QAEvaluator pipeline

### 1. Encoding

- Each (question, answer) pair is tokenized as a text-pair (question as `text`, answer as `text_pair`).
- Sequences are padded / truncated to 512 tokens.

### 2. Scoring

- The sequence-classification model produces logits.
- The positive-class logit is taken as the quality score.
- Higher score → higher predicted quality.

### 3. Ranking

- Indices of the pairs are sorted by descending score.
- `QuestionGenerator` uses these indices to select the best pairs.

## Using a custom fine-tuned evaluator

After running the training script (see [training.md](training.md)) a directory such as `qa-evaluator-finetuned` is created. Point the evaluator to it:

```python
# inside QuestionGenerator.__init__
self.qa_evaluator = QAEvaluator(
    model_name_or_path="/path/to/qa-evaluator-finetuned"
)
```

## Summary

The generator proposes many candidate pairs by treating sentences as answers and asking T5 to invent questions. The evaluator then ranks those pairs so that only the most coherent and answerable ones are kept for the final quiz.
