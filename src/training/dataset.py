"""
Dataset for training the QA Evaluator from a golden list of (question, answer) pairs.

Expected golden format after loading (flat list of dicts):
    [
        {
            "question": "...",
            "answer": "...",          # mapped from ideal_answer
            "context": "...",         # optional, joined from list if present
            "category": "...",        # optional
            "id": "..."               # optional
        },
        ...
    ]

The original nested format (dict of category → list of items) is automatically
flattened by load_golden() in qa_eval_train.py.

Note: the QA Evaluator only sees (question, answer). Context is kept for
possible future use but is ignored during training of the evaluator.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class QAEvalDataset(Dataset):
    """
    Creates positive / negative examples on the fly from a golden QA list.

    - label = 1  → original (question, answer)
    - label = 0  → corrupted (answer swapped or question lightly corrupted)
    """

    def __init__(
        self,
        data: Union[Sequence[Dict[str, str]], Any],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        negative_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data: golden examples. Must contain at least 'question' and 'answer' keys.
            max_length: tokenizer max length (512 is the original setting).
            tokenizer: HuggingFace tokenizer (BERT-style).
            negative_ratio: probability of creating a negative example.
            seed: for reproducibility of the random corruption.
        """
        if hasattr(data, "to_pandas"):
            # HuggingFace Dataset
            self.data = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            self.data = data.reset_index(drop=True)
        else:
            self.data = pd.DataFrame(list(data))

        # Normalise possible key names
        rename_map = {}
        if "ideal_answer" in self.data.columns and "answer" not in self.data.columns:
            rename_map["ideal_answer"] = "answer"
        if "q" in self.data.columns and "question" not in self.data.columns:
            rename_map["q"] = "question"
        if rename_map:
            self.data = self.data.rename(columns=rename_map)

        required = {"question", "answer"}
        if not required.issubset(self.data.columns):
            raise ValueError(
                f"Golden data must contain columns {required}. "
                f"Found: {list(self.data.columns)}"
            )

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.negative_ratio = negative_ratio
        self.rng = random.Random(seed)

        # Keep a plain list of answers for fast random sampling
        self._answers = self.data["answer"].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        row = self.data.iloc[index]
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        # Decide positive vs negative
        is_negative = self.rng.random() < self.negative_ratio
        label = 0 if is_negative else 1

        if is_negative:
            question, answer = self._corrupt(question, answer)

        encoded = self.tokenizer(
            text=question,
            text_pair=answer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get(
                "token_type_ids", torch.zeros_like(encoded["input_ids"])
            ).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Corruption strategies (same spirit as the original repo)
    # ------------------------------------------------------------------
    def _corrupt(self, question: str, answer: str) -> Tuple[str, str]:
        """Randomly choose a corruption method."""
        strategy = self.rng.choice(["swap_answer", "copy_question_fragment"])
        if strategy == "swap_answer":
            return self._swap_answer(question, answer)
        return self._copy_question_fragment(question, answer)

    def _swap_answer(self, question: str, answer: str) -> Tuple[str, str]:
        """Replace the answer with a different golden answer."""
        if len(self._answers) < 2:
            return question, answer + " [corrupted]"

        new_answer = answer
        # Keep sampling until we get a different one
        for _ in range(10):
            candidate = self.rng.choice(self._answers)
            if candidate != answer:
                new_answer = candidate
                break
        return question, new_answer

    def _copy_question_fragment(self, question: str, answer: str) -> Tuple[str, str]:
        """
        Make the answer look like a fragment of the question
        (classic hard-negative for QA evaluators).
        """
        words = question.split()
        if len(words) < 4:
            return self._swap_answer(question, answer)

        # Take a random contiguous span of the question as the new "answer"
        start = self.rng.randint(0, max(0, len(words) - 3))
        end = self.rng.randint(start + 2, min(len(words), start + 8))
        fragment = " ".join(words[start:end])
        return question, fragment
