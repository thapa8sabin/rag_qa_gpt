"""
Simple CPU-friendly Trainer for the QA Evaluator.

Adapted from:
https://github.com/AMontgomerie/question_generator/blob/master/training/trainer.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.training.utils import AverageMeter


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_set: Dataset,
        valid_set: Dataset,
        *,
        device: str = "cpu",
        epochs: int = 5,
        learning_rate: float = 2e-5,
        train_batch_size: int = 8,
        valid_batch_size: int = 16,
        dataloader_workers: int = 0,
        pin_memory: bool = False,
        save_dir: str = "C:\\Users\\User\\Desktop\workspace\\quiz-gpt\\datasets",
        evaluate_on_accuracy: bool = True,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
        )

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.evaluate_on_accuracy = evaluate_on_accuracy

        self.train_loss = AverageMeter()
        self.best_valid_score = 0.0 if evaluate_on_accuracy else float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> None:
        print(f"Training on device: {self.device}")
        print(f"Train batches: {len(self.train_loader)} | Valid batches: {len(self.valid_loader)}")

        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)

            if self.evaluate_on_accuracy:
                score = self.evaluate_accuracy()
                improved = score > self.best_valid_score
                metric_name = "accuracy"
            else:
                score = self.evaluate_loss()
                improved = score < self.best_valid_score
                metric_name = "loss"

            if improved:
                print(
                    f"  → Validation {metric_name} improved "
                    f"({self.best_valid_score:.4f} → {score:.4f}). Saving model."
                )
                self.best_valid_score = score
                self._save()
            else:
                print(f"  → No improvement in validation {metric_name}.")

        print(f"\nBest validation score: {self.best_valid_score:.4f}")
        print(f"Model saved to: {self.save_dir.resolve()}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> None:
        self.model.train()
        self.train_loss.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch")
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.train_loss.update(loss.item(), batch["labels"].size(0))
            pbar.set_postfix({"train_loss": f"{self.train_loss.avg:.4f}"})

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        self.model.eval()
        meter = AverageMeter()
        for batch in tqdm(self.valid_loader, desc="Valid (loss)", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            meter.update(outputs.loss.item(), batch["labels"].size(0))
        print(f"  Valid loss: {meter.avg:.4f}")
        return meter.avg

    @torch.no_grad()
    def evaluate_accuracy(self) -> float:
        self.model.eval()
        all_preds = []
        all_labels = []
        for batch in tqdm(self.valid_loader, desc="Valid (acc)", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        print(f"  Valid accuracy: {acc:.4f}")
        return acc

    def _save(self) -> None:
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
