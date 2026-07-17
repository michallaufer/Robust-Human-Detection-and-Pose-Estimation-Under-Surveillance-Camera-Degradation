#!/usr/bin/env python
"""Validate result tables before treating them as final."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.common.constants import DISTORTION_NAMES, SEVERITIES
from src.common.io import project_path
from src.common.logging_utils import setup_logging
from src.eval.result_schema import metric_ranges_ok


def validate(
    per_image: Path,
    aggregate: Path | None = None,
    expected_rows: int = 11_400,
    expected_groups: int = 76,
    expected_images: int = 150,
) -> list[str]:
    """Validate the final COCO-person result contract."""
    errors: list[str] = []
    if not per_image.exists():
        return [f"Missing {per_image}"]
    df = pd.read_csv(per_image)
    if df.empty:
        return ["Empty per-image table"]
    if len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(df)}")
    if "success" not in df.columns:
        errors.append("Missing success field")
    elif not df["success"].astype(bool).all():
        errors.append(f"Failed rows: {int((~df['success'].astype(bool)).sum())}")

    if "dataset_name" not in df.columns:
        errors.append("Missing dataset_name")
    else:
        names = set(df["dataset_name"].dropna().unique())
        if "demo" in names and per_image.as_posix().find("coco_person") >= 0:
            errors.append("coco_person outputs contain demo rows")
        if len(names) > 1:
            errors.append(f"Mixed dataset_name values: {names}")

    # conditions
    for d in DISTORTION_NAMES:
        for s in SEVERITIES:
            cond = f"{d}_{s}"
            if not ((df["condition"] == cond) | ((df["distortion"] == d) & (df["severity"] == s) & (df["enhanced"] == False))).any():  # noqa: E712
                # soft warning style stored as error for missing distorted
                if not ((df["distortion"] == d) & (df["severity"] == s) & (df["enhanced"] == False)).any():  # noqa: E712
                    errors.append(f"Missing distorted condition: {cond}")
            if not ((df["distortion"] == d) & (df["severity"] == s) & (df["enhanced"] == True)).any():  # noqa: E712
                errors.append(f"Missing enhanced condition: enhanced_{cond}")
    if not (df["condition"] == "clean").any():
        errors.append("Missing clean condition")

    # consistent image ids across conditions for each task
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        id_sets = []
        for cond, g in sub.groupby("condition"):
            id_sets.append((cond, set(g["image_id"].tolist())))
        if id_sets:
            base = id_sets[0][1]
            for cond, ids in id_sets[1:]:
                if ids != base:
                    errors.append(f"Task {task}: image ID set differs for condition {cond}")

    # duplicates
    dup_cols = ["image_id", "task", "condition", "model_name", "weights_path", "enhanced"]
    dup_cols = [c for c in dup_cols if c in df.columns]
    dups = df.duplicated(dup_cols, keep=False)
    if dups.any():
        errors.append(f"Duplicate rows: {int(dups.sum())}")

    group_cols = ["task", "condition", "model_name"]
    group_sizes = df.groupby(group_cols, dropna=False).size()
    if len(group_sizes) != expected_groups:
        errors.append(f"Expected {expected_groups} groups, found {len(group_sizes)}")
    bad_sizes = group_sizes[group_sizes != expected_images]
    if not bad_sizes.empty:
        errors.append(
            f"{len(bad_sizes)} groups do not contain exactly {expected_images} images"
        )

    for _, row in df.iterrows():
        errors.extend([f"range:{x}" for x in metric_ranges_ok(row.to_dict())])

    if aggregate is not None:
        if not aggregate.exists():
            errors.append(f"Missing aggregate table: {aggregate}")
        else:
            agg = pd.read_csv(aggregate)
            if len(agg) != expected_groups:
                errors.append(
                    f"Aggregate expected {expected_groups} rows, found {len(agg)}"
                )
            if "n_images" not in agg.columns:
                errors.append("Aggregate missing n_images")
            elif not agg["n_images"].eq(expected_images).all():
                errors.append(
                    f"Aggregate groups must all have n_images={expected_images}"
                )
            forbidden = [c for c in agg.columns if "ap50" in c.lower()]
            if forbidden:
                errors.append(f"Misleading AP50 columns present: {forbidden}")

    return errors


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--per-image",
        default=str(project_path("results", "coco_person", "tables", "val_per_image.csv")),
    )
    p.add_argument(
        "--aggregate",
        default=str(project_path("results", "coco_person", "tables", "val_aggregate.csv")),
    )
    p.add_argument("--expected-rows", type=int, default=11_400)
    p.add_argument("--expected-groups", type=int, default=76)
    p.add_argument("--expected-images", type=int, default=150)
    args = p.parse_args()
    errs = validate(
        Path(args.per_image),
        aggregate=Path(args.aggregate),
        expected_rows=args.expected_rows,
        expected_groups=args.expected_groups,
        expected_images=args.expected_images,
    )
    if errs:
        logging.error("Validation FAILED (%s issues)", len(errs))
        for e in errs[:50]:
            logging.error(" - %s", e)
        raise SystemExit(1)
    logging.info("Validation OK: %s", args.per_image)


if __name__ == "__main__":
    main()
