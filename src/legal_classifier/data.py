import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import Config


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def build_vocab(texts: List[str], cfg: Config) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        cfg.pad_token: 0,
        cfg.unk_token: 1,
    }

    sorted_tokens = sorted(
        [t for t, c in counter.items() if c >= cfg.min_token_freq],
        key=lambda x: counter[x],
        reverse=True,
    )
    for token in sorted_tokens[: cfg.max_vocab_size - len(vocab)]:
        vocab[token] = len(vocab)

    return vocab


def encode_text(text: str, vocab: Dict[str, int], cfg: Config) -> List[int]:
    tokens = tokenize(text)
    unk_id = vocab[cfg.unk_token]
    ids = [vocab.get(token, unk_id) for token in tokens][: cfg.max_seq_len]
    if len(ids) < cfg.max_seq_len:
        ids += [vocab[cfg.pad_token]] * (cfg.max_seq_len - len(ids))
    return ids


@dataclass
class PreparedData:
    x_train: List[List[int]]
    y_train: List[int]
    x_val: List[List[int]]
    y_val: List[int]
    vocab: Dict[str, int]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]


class LegalDataset(Dataset):
    def __init__(self, x: List[List[int]], y: List[int]) -> None:
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def prepare_data(cfg: Config) -> PreparedData:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.data_path}")

    df = pd.read_csv(cfg.data_path)
    required_cols = {cfg.text_column, cfg.label_column}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {required_cols}. Found: {set(df.columns)}"
        )

    df = df[[cfg.text_column, cfg.label_column]].dropna()
    texts = df[cfg.text_column].astype(str).tolist()
    labels = df[cfg.label_column].astype(str).tolist()

    vocab = build_vocab(texts, cfg)
    label_set = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(label_set)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    x = [encode_text(text, vocab, cfg) for text in texts]
    y = [label_to_id[label] for label in labels]

    if len(x) < 2:
        raise ValueError("Dataset must contain at least 2 rows.")

    stratify_arg = None
    test_size = cfg.val_size

    if len(label_set) > 1:
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())

        # For tiny datasets, enforce enough validation samples for each class
        # when using stratification.
        test_count = max(int(round(len(x) * cfg.val_size)), len(label_set))
        test_count = min(test_count, len(x) - 1)

        if min_class_count >= 2 and test_count >= len(label_set):
            stratify_arg = y
            test_size = test_count

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=cfg.random_seed,
        stratify=stratify_arg,
    )

    return PreparedData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        vocab=vocab,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )


def build_dataloaders(
    prepared: PreparedData, cfg: Config
) -> Tuple[DataLoader, DataLoader]:
    train_ds = LegalDataset(prepared.x_train, prepared.y_train)
    val_ds = LegalDataset(prepared.x_val, prepared.y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
