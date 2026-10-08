# How to use the trained model with your existing QAEvaluator

After running `qa_eval_train.py`, you will have a directory (e.g. `./qa-evaluator-finetuned`) containing:

- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- tokenizer files

## Minimal change to `questiongenerator.py`

In the `QAEvaluator.__init__` method, change the hard-coded path:

```python
# OLD
QAE_PRETRAINED = "iarfmoose/bert-base-cased-qa-evaluator"

# NEW – point to your trained checkpoint
QAE_PRETRAINED = "./qa-evaluator-finetuned"   # or any path / HF model id
```

That is the only change required. The rest of the class (encode / score) stays exactly the same.

## Optional: make the path configurable

If you prefer not to hard-code the path, you can add a constructor argument:

```python
class QAEvaluator:
    def __init__(self, model_name_or_path: str = "iarfmoose/bert-base-cased-qa-evaluator") -> None:
        QAE_PRETRAINED = model_name_or_path
        ...
```

Then in `QuestionGenerator`:

```python
self.qa_evaluator = QAEvaluator(model_name_or_path="./qa-evaluator-finetuned")
```

## Quick smoke-test after training

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("./qa-evaluator-finetuned")
model = AutoModelForSequenceClassification.from_pretrained("./qa-evaluator-finetuned")
model.eval()

def score(q, a):
    enc = tok(q, a, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits
    # higher score on index 1 = more likely to be a good QA pair
    return torch.softmax(logits, dim=-1)[0, 1].item()

print(score("What is the capital of France?", "Paris is the capital of France."))
print(score("What is the capital of France?", "The chemical symbol for gold is Au."))
```

You should see a clearly higher score for the first (golden) pair.
