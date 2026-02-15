import json
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from .config import Config
from .data import build_dataloaders, prepare_data, save_json
from .model import BiLSTMClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    train_mode: bool,
) -> Tuple[float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for x, y in tqdm(dataloader, leave=False):
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_seed)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(cfg)
    train_loader, val_loader = build_dataloaders(prepared, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMClassifier(
        vocab_size=len(prepared.vocab),
        num_classes=len(prepared.label_to_id),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_idx=prepared.vocab[cfg.pad_token],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_acc = 0.0
    best_path = cfg.artifacts_dir / "model.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train_mode=False
        )

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    save_json(prepared.vocab, cfg.artifacts_dir / "vocab.json")
    save_json(prepared.label_to_id, cfg.artifacts_dir / "label_to_id.json")
    save_json(
        {str(k): v for k, v in prepared.id_to_label.items()},
        cfg.artifacts_dir / "id_to_label.json",
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(y.tolist())

    report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
    report_path = cfg.artifacts_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
