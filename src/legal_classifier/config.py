from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_path: Path = Path("data/raw/sample_legal_docs.csv")
    artifacts_dir: Path = Path("artifacts")

    text_column: str = "text"
    label_column: str = "label"

    max_vocab_size: int = 20000
    max_seq_len: int = 256
    min_token_freq: int = 2

    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 1e-3
    val_size: float = 0.2
    random_seed: int = 42

    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3

    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
