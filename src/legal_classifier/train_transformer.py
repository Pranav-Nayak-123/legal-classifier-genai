import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_split(
    texts: List[str], labels: List[int], val_size: float, seed: int
) -> Tuple[List[str], List[str], List[int], List[int]]:
    label_set = set(labels)
    stratify_arg = None
    test_size = val_size
    if len(label_set) > 1:
        counts = {}
        for y in labels:
            counts[y] = counts.get(y, 0) + 1
        min_count = min(counts.values())
        test_count = max(int(round(len(labels) * val_size)), len(label_set))
        test_count = min(test_count, len(labels) - 1)
        if min_count >= 2 and test_count >= len(label_set):
            stratify_arg = labels
            test_size = test_count

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_arg,
    )


def evaluate(model, loader, device) -> Tuple[float, float, Dict]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item() * input_ids.size(0)

            preds = torch.argmax(out.logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return loss, acc, f1, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer legal document classifier.")
    parser.add_argument("--data_path", default="data/raw/legal_scotus.csv")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--model_name", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run: python -m src.legal_classifier.download_dataset"
        )

    df = pd.read_csv(data_path)
    required = {args.text_col, args.label_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain {required}; found {set(df.columns)}")

    df = df[[args.text_col, args.label_col]].dropna()
    texts = df[args.text_col].astype(str).tolist()
    raw_labels = df[args.label_col].astype(str).tolist()
    if len(texts) < 2:
        raise ValueError("Dataset too small. Need at least 2 rows.")

    label_to_id = {label: i for i, label in enumerate(sorted(set(raw_labels)))}
    id_to_label = {i: label for label, i in label_to_id.items()}
    labels = [label_to_id[x] for x in raw_labels]

    x_train, x_val, y_train, y_val = safe_split(texts, labels, args.val_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = TextClassificationDataset(x_train, y_train, tokenizer, args.max_len)
    val_ds = TextClassificationDataset(x_val, y_val, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_to_id)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    artifacts_dir = Path(args.artifacts_dir)
    model_dir = artifacts_dir / "transformer_model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        y_true = []
        y_pred = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor,
            )
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(out.logits, dim=1)
            y_true.extend(labels_tensor.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        val_loss, val_acc, val_f1, val_report = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            with (artifacts_dir / "classification_report.json").open("w", encoding="utf-8") as f:
                json.dump(val_report, f, indent=2)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    with (artifacts_dir / "label_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2)
    with (artifacts_dir / "id_to_label.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2)
    with (artifacts_dir / "model_type.json").open("w", encoding="utf-8") as f:
        json.dump({"type": "transformer", "model_name": args.model_name}, f, indent=2)
    with (artifacts_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best transformer model to {model_dir}")
    print(f"Best validation macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

