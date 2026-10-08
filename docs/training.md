# Training the Custom QA Evaluator

The quality of generated quizzes depends heavily on the ranking model. Quiz-GPT therefore fine-tunes a sequence-classification model on a mixture of a hand-crafted **golden dataset** and public Hugging Face data.

## Data sources

### 1. Golden dataset

Located at `golden/golden_dataset.json` (or a path supplied via CLI).

Structure (simplified):

```json
{
  "arts_and_literature": [
    {
      "id": "q1",
      "question": "True or false, The Harry Potter series was written by Lewis Carroll.",
      "context": ["..."],
      "ideal_answer": "False"
    },
    ...
  ],
  "general_knowledge": [ ... ],
  "geography": [ ... ],
  "science_and_nature": [ ... ],
  "sports_and_pop_culture": [ ... ]
}
```

Each entry supplies a reliable **positive** example (question + ideal answer that is supported by the given context).

### 2. Public Hugging Face dataset

By default the script can load `iarfmoose/qa_evaluator` (or a similar public QA-evaluation corpus). A subsample is taken to keep training feasible on CPU.

## Creating training examples

- **Positive pairs** (label = 1)  
  Original (question, ideal_answer) pairs from both the golden set and the public set.

- **Negative pairs** (label = 0)  
  Created by corrupting the positives, for example:
  - swapping the answer with an answer from a different question,
  - using a fragment of the question itself as the answer,
  - pairing a question with unrelated context.

The golden examples are usually **upsampled** so that the relatively small domain-specific set is not drowned by the public data.

## Running the training script

```bash
uv run python -m src.training.qa_eval_train \
  --golden_path golden/golden_dataset.json \
  --use_public \
  --public_sample_size 5000 \
  --golden_upsample 20 \
  --device cpu \
  --epochs 3 \
  --train_batch_size 8 \
  --valid_batch_size 16 \
  --learning_rate 2e-5 \
  --valid_ratio 0.05
```

### Important flags

| Flag                  | Meaning                                      | Suggested value      |
|-----------------------|----------------------------------------------|----------------------|
| `--use_public`        | Load the public Hugging Face dataset         | always enabled       |
| `--public_sample_size`| Number of public examples to keep            | 3000–8000 (CPU)      |
| `--golden_upsample`   | How many times each golden example is repeated | 10–30              |
| `--valid_ratio`       | Fraction held out for validation             | 0.05                 |
| `--device`             | `cpu` or `cuda`                              | `cpu` for most users |
| `--epochs`            | Number of training epochs                    | 3–5                  |

With the numbers above you obtain roughly:

- 20 golden × 20 upsample ≈ 400 domain examples
- 5000 public examples
- → ~5400 total → train ≈ 5100 / valid ≈ 270

This is sufficient for the model to learn useful ranking behaviour while remaining practical on CPU.

## The Trainer class

File: `src/training/trainer.py`

Responsibilities:

- Accepts a pre-trained model, tokenizer, train set and validation set.
- Creates DataLoaders (configurable batch sizes, workers, pin memory).
- Uses **AdamW** with the supplied learning rate.
- Applies gradient clipping (`max_grad_norm`).
- For each epoch:
  1. Trains one full epoch, tracking average loss.
  2. Evaluates on the validation set (accuracy by default, or loss).
  3. If the validation metric improved, saves both model and tokenizer with `save_pretrained`.
- At the end reports the best validation score and the save directory.

Evaluation modes:

- **Accuracy** (default) – `torch.argmax` on logits + `sklearn.metrics.accuracy_score`.
- **Loss** – average of the model’s own loss.

## After training

A directory (e.g. `qa-evaluator-finetuned`) is created containing:

- `config.json`
- model weights (`pytorch_model.bin` or `model.safetensors`)
- tokenizer files

Point `QAEvaluator` at this directory (see [question-generation.md](question-generation.md)).

## Summary of the training philosophy

| Goal                         | How it is achieved                                      |
|------------------------------|---------------------------------------------------------|
| Domain relevance             | Golden dataset of carefully written educational Q-A pairs |
| Scale & robustness           | Public Hugging Face QA data                             |
| Class balance                | Upsampling of golden examples + explicit negative construction |
| Practical training           | Small batches, CPU-friendly defaults, early best-model saving |

The resulting fine-tuned classifier becomes the quality gate that decides which generated questions are good enough to present to the user.
