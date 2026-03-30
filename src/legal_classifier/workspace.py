import json
from datetime import datetime
from pathlib import Path
from typing import Dict


WORKSPACE_DIR = Path("workspace_cases")


def save_workspace(payload: Dict) -> Path:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = WORKSPACE_DIR / f"case_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path

