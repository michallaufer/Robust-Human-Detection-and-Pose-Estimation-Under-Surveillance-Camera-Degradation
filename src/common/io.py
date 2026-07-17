"""Path and YAML I/O helpers."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.common.constants import ROOT


def project_path(*parts: str | Path) -> Path:
    return ROOT.joinpath(*[str(p) for p in parts])


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = project_path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML: {path}")
    return data


def save_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = project_path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + replace for crash safety."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def atomic_write_csv(path: Path, df) -> None:
    """Pandas DataFrame to CSV atomically."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
        newline="",
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    # pandas may write before flush close; reopen via path
    # Re-write properly:
    tmp_csv = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(path)
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
