import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .data import encode_text
from .model import BiLSTMClassifier

SCOTUS_ISSUE_AREA_MAP = {
    "1": "Criminal Procedure",
    "2": "Civil Rights",
    "3": "First Amendment",
    "4": "Due Process",
    "5": "Privacy",
    "6": "Attorneys",
    "7": "Unions",
    "8": "Economic Activity",
    "9": "Judicial Power",
    "10": "Federalism",
    "11": "Interstate Relations",
    "12": "Federal Taxation",
    "13": "Miscellaneous",
    "14": "Private Action",
}


class Predictor:
    def __init__(self, artifacts_dir: Path = Path("artifacts")) -> None:
        self.cfg = Config()
        self.artifacts_dir = artifacts_dir
        self.model_type = self._detect_model_type()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_to_id = self._load_json("label_to_id.json")
        raw_id_to_label = self._load_json("id_to_label.json")
        self.id_to_label = {int(k): v for k, v in raw_id_to_label.items()}

        if self.model_type == "transformer":
            self._load_transformer_model()
        else:
            self._load_bilstm_model()

    def _load_json(self, name: str) -> Dict:
        path = self.artifacts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _detect_model_type(self) -> str:
        model_type_path = self.artifacts_dir / "model_type.json"
        if model_type_path.exists():
            with model_type_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("type", "bilstm")
        return "bilstm"

    def _load_bilstm_model(self) -> None:
        self.vocab = self._load_json("vocab.json")
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            num_classes=len(self.label_to_id),
            embed_dim=self.cfg.embed_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            pad_idx=self.vocab[self.cfg.pad_token],
        ).to(self.device)

        model_path = self.artifacts_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                "Trained BiLSTM model not found. Run: python -m src.legal_classifier.train"
            )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _load_transformer_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers package is missing. Run: pip install -r requirements.txt"
            ) from exc

        model_dir = self.artifacts_dir / "transformer_model"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Transformer model directory not found: {model_dir}. "
                "Run: python -m src.legal_classifier.train_transformer"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> Tuple[str, float, List[Dict]]:
        if self.model_type == "transformer":
            probs = self._predict_transformer_probs(text)
        else:
            probs = self._predict_bilstm_probs(text)

        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.numel()))
        predictions = []
        for p, i in zip(top_probs.tolist(), top_ids.tolist()):
            raw_label = self.id_to_label[int(i)]
            predictions.append(
                {
                    "label": self._display_label(raw_label),
                    "confidence": float(p),
                }
            )

        return predictions[0]["label"], predictions[0]["confidence"], predictions

    def _predict_bilstm_probs(self, text: str) -> torch.Tensor:
        encoded = encode_text(text, self.vocab, self.cfg)
        x = torch.tensor([encoded], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            return F.softmax(logits, dim=1).squeeze(0)

    def _predict_transformer_probs(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return F.softmax(out.logits, dim=1).squeeze(0)

    def _display_label(self, label: str) -> str:
        label_str = str(label).strip()
        return SCOTUS_ISSUE_AREA_MAP.get(label_str, label_str)
