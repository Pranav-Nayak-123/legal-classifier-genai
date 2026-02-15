import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _resolve_text_field(example: Dict) -> Optional[str]:
    for candidate in ("text", "document", "content", "sentence"):
        if candidate in example:
            value = example[candidate]
            if isinstance(value, list):
                return " ".join(str(x) for x in value)
            return str(value)

    for key, value in example.items():
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return " ".join(value)
    return None


def _build_rows(dataset, split_name: str) -> List[Dict]:
    rows: List[Dict] = []
    label_names = None
    if "label" in dataset.features and hasattr(dataset.features["label"], "names"):
        label_names = dataset.features["label"].names

    for ex in dataset:
        text = _resolve_text_field(ex)
        if not text:
            continue

        label_value = ex.get("label")
        if isinstance(label_value, int) and label_names and 0 <= label_value < len(label_names):
            label = label_names[label_value]
        else:
            label = str(label_value)

        rows.append({"text": text, "label": label, "split": split_name})

    return rows


def download_and_export(
    dataset_name: str,
    subset: str,
    output_path: Path,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    ds = load_dataset(dataset_name, subset)

    all_rows: List[Dict] = []
    for split_name in ds.keys():
        all_rows.extend(_build_rows(ds[split_name], split_name))

    if not all_rows:
        raise ValueError("No rows were extracted from the dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    print(f"Saved {len(all_rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download legal dataset and export CSV.")
    parser.add_argument("--dataset", default="lex_glue", help="Hugging Face dataset name")
    parser.add_argument("--subset", default="scotus", help="Dataset config/subset")
    parser.add_argument(
        "--output",
        default="data/raw/legal_scotus.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_export(args.dataset, args.subset, Path(args.output))


if __name__ == "__main__":
    main()

