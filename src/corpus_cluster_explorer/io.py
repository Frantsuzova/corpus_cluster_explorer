import json
import os
from typing import Any

import numpy as np
import pandas as pd

from .config import DATE_PATTERNS, TEXT_FIELDS_PRIORITY


def load_structured_dataset(path: str) -> list[dict[str, Any]]:
    """
    Поддерживаемые форматы:
    - .jsonl
    - .json
    - .csv
    - .tsv
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        return load_jsonl(path)

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return [x for x in data["data"] if isinstance(x, dict)]
            return [data]

        raise ValueError("JSON-файл не содержит список объектов или словарь.")

    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        return df.to_dict(orient="records")

    raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_tokenized_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "tokens" not in row:
                raise ValueError("В токенизированном JSONL отсутствует поле 'tokens'.")
            rows.append(row)
    return rows


def looks_like_date(s: str) -> bool:
    s = s.strip()
    return any(p.match(s) for p in DATE_PATTERNS)


def looks_like_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("www.")


def looks_like_text_series(values: list[str]) -> bool:
    vals = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return False

    date_ratio = sum(looks_like_date(v) for v in vals) / len(vals)
    if date_ratio > 0.7:
        return False

    url_ratio = sum(looks_like_url(v) for v in vals) / len(vals)
    if url_ratio > 0.7:
        return False

    avg_len = float(np.mean([len(v) for v in vals]))
    if avg_len < 20:
        return False

    unique_ratio = len(set(vals)) / len(vals)
    if unique_ratio < 0.1:
        return False

    return True


def detect_text_fields(records: list[dict[str, Any]], sample_size: int = 200) -> list[str]:
    if not records:
        return []

    sample = records[:sample_size]
    all_keys: set[str] = set()
    for rec in sample:
        all_keys.update(rec.keys())

    candidates: list[str] = []
    for key in all_keys:
        values = [rec.get(key) for rec in sample if isinstance(rec.get(key), str)]
        if looks_like_text_series(values):
            candidates.append(key)

    priority = [f for f in TEXT_FIELDS_PRIORITY if f in candidates]
    rest = [f for f in candidates if f not in priority]
    return priority + sorted(rest)


def extract_nested_strings(value: Any) -> list[str]:
    texts: list[str] = []

    if isinstance(value, str):
        if value.strip():
            texts.append(value.strip())
        return texts

    if isinstance(value, dict):
        for key in TEXT_FIELDS_PRIORITY:
            if key in value and isinstance(value[key], str) and value[key].strip():
                texts.append(value[key].strip())

        for _, v in value.items():
            if isinstance(v, (dict, list)):
                texts.extend(extract_nested_strings(v))
        return texts

    if isinstance(value, list):
        for item in value:
            texts.extend(extract_nested_strings(item))
        return texts

    return texts


def build_analysis_documents(
    records: list[dict[str, Any]],
    selected_fields: list[str],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for rec in records:
        field_text_map: dict[str, str] = {}

        for field in selected_fields:
            texts = extract_nested_strings(rec.get(field))
            field_text_map[field] = " ".join([t for t in texts if t.strip()])

        combined = " ".join(
            [field_text_map[f] for f in selected_fields if field_text_map[f].strip()]
        ).strip()

        docs.append({
            "field_text_map": field_text_map,
            "combined_text": combined
        })

    return docs
