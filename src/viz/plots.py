"""Publication-style matplotlib plots (no seaborn)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.constants import DISTORTION_NAMES, SEVERITIES
from src.common.io import ensure_dir, project_path

logger = logging.getLogger(__name__)
SEV_ORDER = ["clean", "mild", "medium", "severe"]


def _load_agg(path: Path | None = None) -> pd.DataFrame:
    path = path or project_path("results", "coco_person", "tables", "val_aggregate.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate table: {path}")
    df = pd.read_csv(path)
    if "dataset_name" in df.columns and (df.get("dataset_name") == "demo").any():
        raise ValueError("Refuse to plot demo rows as coco_person results")
    return df


def _series(
    df: pd.DataFrame,
    task: str,
    metric: str,
    distortion: str,
    enhanced: bool,
    model_name: str | None = None,
) -> list[tuple[str, float]]:
    pts = []
    task_df = df[df["task"] == task]
    if model_name is not None:
        task_df = task_df[task_df["model_name"] == model_name]
    # clean once
    clean = task_df[task_df["condition"] == "clean"]
    if len(clean) and metric in clean.columns:
        pts.append(("clean", float(clean.iloc[0][metric])))
    for sev in SEVERITIES:
        if enhanced:
            sub = task_df[
                (task_df["distortion"] == distortion)
                & (task_df["severity"] == sev)
                & (task_df["enhanced"] == True)  # noqa: E712
            ]
        else:
            sub = task_df[
                (task_df["distortion"] == distortion)
                & (task_df["severity"] == sev)
                & (task_df["enhanced"] == False)  # noqa: E712
            ]
        if len(sub) and metric in sub.columns:
            pts.append((sev, float(sub.iloc[0][metric])))
    return pts


def plot_metric_vs_severity(
    df: pd.DataFrame,
    task: str,
    metric: str,
    title: str,
    out: Path,
    model_name: str | None = None,
) -> Path:
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    records = []
    for dist in DISTORTION_NAMES:
        for enh, style in [(False, "-o"), (True, "--s")]:
            pts = _series(df, task, metric, dist, enh, model_name=model_name)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label = f"{dist}/{'enh' if enh else 'dist'}"
            ax.plot(xs, ys, style, label=label)
            records.extend(
                {
                    "task": task,
                    "model_name": model_name or "",
                    "distortion": dist,
                    "input": "enhanced" if enh else "distorted",
                    "severity": severity,
                    metric: value,
                }
                for severity, value in pts
            )
    ax.set_title(title)
    ax.set_xlabel("Severity")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    pd.DataFrame(records).to_csv(out.with_suffix(".csv"), index=False)
    plt.close(fig)
    return out


def plot_pose_comparison(df: pd.DataFrame, out: Path) -> Path:
    """Compare pretrained and fine-tuned pose PCK for raw/enhanced inputs."""
    ensure_dir(out.parent)
    models = {
        "yolov8n-pose": "Pretrained",
        "pose_robust_ft_best": "Fine-tuned",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    records = []
    for ax, dist in zip(axes, DISTORTION_NAMES):
        for model, model_label in models.items():
            for enhanced, style in [(False, "-o"), (True, "--s")]:
                pts = _series(
                    df,
                    "pose",
                    "pck_02",
                    dist,
                    enhanced,
                    model_name=model,
                )
                if not pts:
                    continue
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    style,
                    label=f"{model_label}/{'enh' if enhanced else 'dist'}",
                )
                records.extend(
                    {
                        "task": "pose",
                        "model_name": model,
                        "distortion": dist,
                        "input": "enhanced" if enhanced else "distorted",
                        "severity": severity,
                        "pck_02": value,
                    }
                    for severity, value in pts
                )
        ax.set_title(dist.replace("_", " ").title())
        ax.set_xlabel("Severity")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PCK@0.2")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Pose robustness: pretrained vs fine-tuned")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    pd.DataFrame(records).to_csv(out.with_suffix(".csv"), index=False)
    plt.close(fig)
    return out


def plot_psnr_scatter(df: pd.DataFrame, task: str, metric: str, out: Path) -> Path:
    ensure_dir(out.parent)
    sub = df[(df["task"] == task) & (df["condition"] != "clean")].copy()
    if task == "pose":
        sub = sub[sub["model_name"] == "yolov8n-pose"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["psnr"], sub[metric], alpha=0.8)
    ax.set_xlabel("Mean PSNR vs clean (dB)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs PSNR ({task})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    sub[["condition", "model_name", "psnr", metric]].to_csv(
        out.with_suffix(".csv"), index=False
    )
    plt.close(fig)
    return out


def plot_boundary_vs_detection(df: pd.DataFrame, out: Path) -> Path:
    b = df[df["task"] == "boundary"][["condition", "boundary_f1"]].rename(columns={"boundary_f1": "f1"})
    d = df[df["task"] == "detection"][["condition", "recall"]].rename(columns={"recall": "det_recall"})
    m = b.merge(d, on="condition")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["f1"], m["det_recall"])
    for _, r in m.iterrows():
        ax.annotate(str(r["condition"]), (r["f1"], r["det_recall"]), fontsize=6, alpha=0.7)
    ax.set_xlabel("Boundary F1")
    ax.set_ylabel("Detection recall")
    ax.set_title("Boundary F1 vs detection recall")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    m.to_csv(out.with_suffix(".csv"), index=False)
    plt.close(fig)
    return out


def make_all_plots(aggregate_csv: Path | None = None) -> list[Path]:
    df = _load_agg(aggregate_csv)
    fig_dir = ensure_dir(project_path("results", "coco_person", "figures"))
    outs = []
    outs.append(
        plot_metric_vs_severity(
            df,
            "boundary",
            "boundary_f1",
            "Boundary F1 vs severity",
            fig_dir / "boundary_f1_vs_severity",
            model_name="canny",
        )
    )
    if "recall" in df.columns:
        outs.append(
            plot_metric_vs_severity(
                df,
                "detection",
                "recall",
                "Detection recall vs severity",
                fig_dir / "det_recall_vs_severity",
                model_name="yolov8n",
            )
        )
        outs.append(
            plot_metric_vs_severity(
                df,
                "detection",
                "f1",
                "Detection F1 vs severity",
                fig_dir / "det_f1_vs_severity",
                model_name="yolov8n",
            )
        )
    if "pck_02" in df.columns:
        outs.append(plot_pose_comparison(df, fig_dir / "pose_pck_vs_severity"))
    if "psnr" in df.columns and "pck_02" in df.columns:
        outs.append(plot_psnr_scatter(df, "pose", "pck_02", fig_dir / "pose_pck_vs_psnr"))
    outs.append(plot_boundary_vs_detection(df, fig_dir / "boundary_vs_detection"))
    logger.info("Wrote %s figures", len(outs))
    return outs
