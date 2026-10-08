#!/usr/bin/env python
"""
Train (or fine-tune) a QA Evaluator on your golden dataset,
optionally mixed with the public iarfmoose/qa_evaluator dataset.

Recommended CPU-friendly command (mixes golden + public subsample):
---------------------------------------------------------------
python qa_eval_train.py \
    --golden_path /path/to/golden_dataset.json \
    --use_public \
    --public_sample_size 5000 \
    --golden_upsample 20 \
    --save_dir ./qa-evaluator-finetuned \
    --device cpu \
    --epochs 3 \
    --train_batch_size 8 \
    --valid_batch_size 16

After training, point QAEvaluator to the save_dir.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.training.dataset import QAEvalDataset
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QA Evaluator on golden + optional public data (CPU-friendly)"
    )
    parser.add_argument(
        "--golden_path",
        type=str,
        default=None,
        help="Path to golden QA file (json/jsonl/csv). Optional if --use_public only.",
    )
    parser.add_argument(
        "--use_public",
        action="store_true",
        help="Also load the public HuggingFace dataset iarfmoose/qa_evaluator",
    )
    parser.add_argument(
        "--public_sample_size",
        type=int,
        default=5000,
        help="How many public examples to keep (subsample for CPU). 0 = use all.",
    )
    parser.add_argument(
        "--golden_upsample",
        type=int,
        default=10,
        help="Repeat each golden example this many times so the model sees your domain more often.",
    )
    parser.add_argument("--save_dir", type=str, default="./qa-evaluator-finetuned")
    parser.add_argument("--base_model", type=str, default="bert-base-cased")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.05,
        help="Fraction of the *combined* data held out for validation",
    )
    parser.add_argument("--negative_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_workers", type=int, default=0)
    return parser.parse_args()


def _flatten_category_dict(data: dict) -> List[dict]:
    """
    Flatten the nested golden format:

        {
          "category_name": [
            {
              "id": "...",
              "question": "...",
              "context": ["..."],
              "ideal_answer": "..."
            },
            ...
          ],
          ...
        }

    into a flat list of {"question", "answer", "context", "category", "id"}.
    """
    rows = []
    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            question = item.get("question") or item.get("q")
            answer = (
                item.get("ideal_answer")
                or item.get("answer")
                or item.get("gold_answer")
            )
            if not question or not answer:
                continue

            # context may be a list of strings or a single string
            ctx = item.get("context", [])
            if isinstance(ctx, list):
                context = " ".join(str(c).strip() for c in ctx if c)
            else:
                context = str(ctx).strip() if ctx else ""

            rows.append(
                {
                    "id": item.get("id", ""),
                    "category": category,
                    "question": str(question).strip(),
                    "answer": str(answer).strip(),
                    "context": context,
                }
            )
    return rows


def load_golden(path: str) -> List[dict]:
    """Load golden data from several common formats."""
    p = Path(path)

    # Try HuggingFace datasets first (if the string looks like a dataset id)
    if not p.exists() and "/" in path:
        try:
            import datasets
            ds = datasets.load_dataset(path)
            # Prefer 'train' split, otherwise take the first available
            split = "train" if "train" in ds else list(ds.keys())[0]
            print(f"Loaded HuggingFace dataset '{path}' split='{split}'")
            return list(ds[split])
        except Exception as e:
            raise FileNotFoundError(
                f"Path '{path}' does not exist and could not be loaded as HF dataset: {e}"
            )

    if not p.exists():
        raise FileNotFoundError(f"Golden file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: nested category dict (user's golden_dataset.json format)
        if isinstance(data, dict):
            # common wrapper key
            if "data" in data and isinstance(data["data"], (list, dict)):
                data = data["data"]

            if isinstance(data, dict):
                # Heuristic: values are lists of dicts that look like QA items
                sample_val = next(iter(data.values()), None)
                if isinstance(sample_val, list) and sample_val and isinstance(sample_val[0], dict):
                    if any(k in sample_val[0] for k in ("ideal_answer", "answer", "question")):
                        flattened = _flatten_category_dict(data)
                        print(f"  Detected nested category format → flattened to {len(flattened)} examples")
                        return flattened

        # Case 2: already a flat list
        if isinstance(data, list):
            # Normalise keys if needed
            normalised = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = item.get("question") or item.get("q")
                a = item.get("ideal_answer") or item.get("answer") or item.get("gold_answer")
                if q and a:
                    normalised.append({
                        "question": str(q).strip(),
                        "answer": str(a).strip(),
                        "context": item.get("context", ""),
                        "id": item.get("id", ""),
                        "category": item.get("category", ""),
                    })
            return normalised

        raise ValueError(
            "JSON root must be either a list of QA objects or a dict of category → list of QA objects"
        )

    if suffix == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported file type: {suffix}. Use .json, .jsonl, .csv or .tsv")


def train_val_split(data: List[dict], valid_ratio: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_valid = max(1, int(len(data) * valid_ratio)) if len(data) > 1 else 0
    valid_idx = set(indices[:n_valid])
    train = [data[i] for i in indices if i not in valid_idx]
    valid = [data[i] for i in indices if i in valid_idx]
    return train, valid


def load_public(sample_size: int, seed: int) -> List[dict]:
    """Load and optionally subsample the public iarfmoose/qa_evaluator dataset."""
    import datasets

    print("Loading public dataset: iarfmoose/qa_evaluator ...")
    ds = datasets.load_dataset("iarfmoose/qa_evaluator", split="train")
    print(f"  → public train size: {len(ds)}")

    if sample_size > 0 and sample_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(sample_size))
        print(f"  → subsampled to {sample_size} examples")

    rows = []
    for item in ds:
        q = item.get("question")
        a = item.get("answer")
        if q and a:
            rows.append(
                {
                    "question": str(q).strip(),
                    "answer": str(a).strip(),
                    "context": "",
                    "id": "",
                    "category": "public",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.golden_path and not args.use_public:
        raise SystemExit("Provide at least --golden_path or --use_public")

    combined: List[dict] = []

    # ----- golden data -----
    if args.golden_path:
        print("Loading golden data...")
        golden = load_golden(args.golden_path)
        print(f"  → {len(golden)} golden examples")

        if args.golden_upsample > 1:
            golden = golden * args.golden_upsample
            print(f"  → upsampled golden to {len(golden)} (x{args.golden_upsample})")

        combined.extend(golden)
    else:
        print("No golden_path provided – using public data only")

    # ----- public data -----
    if args.use_public:
        public = load_public(args.public_sample_size, args.seed)
        combined.extend(public)

    if not combined:
        raise SystemExit("No training data loaded")

    print(f"\nCombined dataset size: {len(combined)}")

    train_data, valid_data = train_val_split(combined, args.valid_ratio, args.seed)
    print(f"  → train: {len(train_data)} | valid: {len(valid_data)}")

    print(f"\nLoading tokenizer & model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
    )

    train_set = QAEvalDataset(
        train_data,
        max_length=args.max_length,
        tokenizer=tokenizer,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    valid_set = QAEvalDataset(
        valid_data,
        max_length=args.max_length,
        tokenizer=tokenizer,
        negative_ratio=args.negative_ratio,
        seed=args.seed + 1,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        dataloader_workers=args.dataloader_workers,
        save_dir=args.save_dir,
        evaluate_on_accuracy=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()

    