from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .benchmarking import load_benchmark_config
from .utils import ensure_dir, write_json

try:  # pragma: no cover - optional dependency is already available in the workspace
    import umap
except Exception:  # pragma: no cover - graceful fallback when UMAP is unavailable
    umap = None


sns.set_theme(style="whitegrid", context="talk")


def generate_run_visualizations(
    result_dir: Path,
    *,
    device: str | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    vis_dir = ensure_dir(result_dir / "visualizations")
    summary: dict[str, Any] = {
        "result_dir": str(result_dir),
        "visualization_dir": str(vis_dir),
        "generated": [],
        "skipped": [],
        "errors": [],
    }

    config = _load_config(result_dir)
    task_kind = _infer_task_kind(result_dir, config)
    prediction_path = _pick_prediction_path(result_dir)

    if (result_dir / "training_history.csv").exists():
        _run_step(summary, "training_history", _plot_training_history, result_dir, vis_dir, overwrite=overwrite)

    if (result_dir / "metrics.csv").exists() or (result_dir / "ablation_summary.csv").exists():
        _run_step(summary, "metrics_overview", _plot_metrics_overview, result_dir, vis_dir, task_kind, overwrite=overwrite)

    if (result_dir / "gpu_runtime.json").exists():
        _run_step(summary, "runtime_breakdown", _plot_runtime_breakdown, result_dir, vis_dir, overwrite=overwrite)

    if prediction_path is not None:
        _run_step(summary, "prediction_diagnostics", _plot_prediction_diagnostics, result_dir, prediction_path, vis_dir, overwrite=overwrite)

    if (result_dir / "ablation_summary.csv").exists():
        _run_step(summary, "ablation_summary", _plot_ablation_summary, result_dir, vis_dir, overwrite=overwrite)

    latent_path = _ensure_terminal_latents(
        result_dir,
        config=config,
        prediction_path=prediction_path,
        device=device,
        overwrite=overwrite,
        summary=summary,
    )
    if latent_path is not None:
        _run_step(summary, "latent_projection", _plot_latent_projection, result_dir, prediction_path, latent_path, vis_dir, overwrite=overwrite)

    _write_index_files(result_dir, vis_dir, summary)
    return summary


def generate_workspace_visualizations(
    root: Path,
    *,
    device: str | None = None,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    summaries: list[dict[str, Any]] = []
    for benchmark_dir in _iter_benchmark_dirs(root):
        summaries.extend(generate_benchmark_visualizations(benchmark_dir, device=device, overwrite=overwrite))
    return summaries


def generate_benchmark_visualizations(
    benchmark_dir: Path,
    *,
    device: str | None = None,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    benchmark_dir = Path(benchmark_dir).resolve()
    results_root = benchmark_dir / "results"
    if not results_root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for result_dir in _iter_result_dirs(results_root):
        summaries.append(generate_run_visualizations(result_dir, device=device, overwrite=overwrite))
    return summaries


def _iter_benchmark_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and (child / "results").exists():
            yield child


def _iter_result_dirs(results_root: Path) -> Iterable[Path]:
    results_root = Path(results_root)
    if _is_result_dir(results_root):
        yield results_root
    for path in sorted(results_root.rglob("*")):
        if path.is_dir() and _is_result_dir(path):
            yield path


def _is_result_dir(path: Path) -> bool:
    if path.name == "visualizations" or path.name.startswith("."):
        return False
    excluded_parts = {".cache", "visualizations"}
    if any(part in excluded_parts for part in path.parts):
        return False
    indicators = [
        "training_history.csv",
        "predictions.csv",
        "metrics.csv",
        "gdsc_metrics.csv",
        "tcga_binary_metrics.csv",
        "tcga_actual_treatment_scores.csv",
        "ablation_summary.csv",
        "gpu_runtime.json",
    ]
    return any((path / name).exists() for name in indicators)


def _load_config(result_dir: Path):
    config_path = result_dir / "default.yaml"
    if config_path.exists():
        try:
            return load_benchmark_config(config_path)
        except Exception:
            return None
    return None


def _infer_task_kind(result_dir: Path, config) -> str:
    if config is not None and getattr(config, "task_type", None) == "tcga_binary_response":
        return "tcga_binary"
    if (result_dir / "tcga_binary_metrics.csv").exists() or (result_dir / "tcga_binary_scores.csv").exists():
        return "tcga_binary"
    if (result_dir / "tcga_actual_treatment_scores.csv").exists():
        return "tcga_actual_treatment"
    if (result_dir / "ctrdb_response_metrics.csv").exists() or (result_dir / "ctrdb_response_scores.csv").exists():
        return "ctrdb"
    return "gdsc"


def _pick_prediction_path(result_dir: Path) -> Path | None:
    for name in [
        "predictions.csv",
        "tcga_binary_scores.csv",
        "tcga_actual_treatment_scores.csv",
        "ctrdb_response_scores.csv",
    ]:
        candidate = result_dir / name
        if candidate.exists():
            return candidate
    return None


def _run_step(summary: dict[str, Any], name: str, fn, *args, overwrite: bool = True, **kwargs) -> None:
    try:
        outputs = fn(*args, overwrite=overwrite, **kwargs)
    except Exception as exc:  # pragma: no cover - batch backfills should keep going
        summary["errors"].append({"step": name, "error": f"{type(exc).__name__}: {exc}"})
        return
    if not outputs:
        summary["skipped"].append(name)
        return
    if isinstance(outputs, (list, tuple)):
        summary["generated"].extend(str(x) for x in outputs)
    else:
        summary["generated"].append(str(outputs))


def _save_figure(fig: plt.Figure, path: Path, *, overwrite: bool = True) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        plt.close(fig)
        return path
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _plot_training_history(result_dir: Path, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    history = _safe_read_csv(result_dir / "training_history.csv")
    if history.empty:
        return []
    epoch_col = "epoch" if "epoch" in history.columns else None
    value_cols = [
        col
        for col in [
            "train_loss",
            "loss_total",
            "val_loss",
            "loss_raw",
            "loss_absolute_auc",
            "loss_value",
            "loss_drug_centered",
            "loss_cell_residual",
            "loss_rank_drug",
            "loss_same_drug_cross_cell_rank",
            "loss_adv",
            "loss_same_cell_cross_drug_rank",
            "loss_graph_consistency",
            "train_bce",
            "val_bce",
            "val_cell_residual_loss",
        ]
        if col in history.columns
    ]
    if not value_cols:
        return []
    fig, ax = plt.subplots(figsize=(11, 6))
    x = history[epoch_col] if epoch_col else np.arange(1, len(history) + 1)
    for col in value_cols:
        ax.plot(x, history[col], marker="o", linewidth=2, label=col)
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / Metric")
    ax.legend(loc="best", frameon=True)
    return [_save_figure(fig, vis_dir / "training_history.png", overwrite=overwrite)]


def _metric_x_column(frame: pd.DataFrame) -> str:
    for candidate in ["split", "metric_source", "analysis", "variant"]:
        if candidate in frame.columns:
            return candidate
    return frame.columns[0]


def _plot_metrics_overview(result_dir: Path, vis_dir: Path, task_kind: str, *, overwrite: bool = True) -> list[Path]:
    if (result_dir / "ablation_summary.csv").exists():
        frame = _safe_read_csv(result_dir / "ablation_summary.csv")
    else:
        frame = _safe_read_csv(result_dir / "metrics.csv")
    if frame.empty:
        return []

    x_col = _metric_x_column(frame)
    if task_kind == "tcga_binary":
        candidates = ["auroc", "auprc", "balanced_accuracy", "brier", "accuracy", "f1", "precision", "recall"]
    else:
        candidates = [
            "overall_pcc",
            "within_drug_pcc_mean",
            "within_cell_pcc_mean",
            "value_spearman",
            "raw_auc_rmse",
            "value_rmse",
        ]
    metric_cols = [col for col in candidates if col in frame.columns]
    if not metric_cols:
        metric_cols = [
            col
            for col in frame.columns
            if col != x_col and pd.api.types.is_numeric_dtype(frame[col])
            and col not in {"n", "threshold", "n_pos", "n_neg"}
        ][:4]
    if not metric_cols:
        return []

    fig, axes = plt.subplots(len(metric_cols), 1, figsize=(11, 4 * len(metric_cols)), squeeze=False)
    for ax, metric in zip(axes.ravel(), metric_cols):
        plot_frame = frame[[x_col, metric]].dropna()
        if plot_frame.empty:
            ax.axis("off")
            continue
        sns.barplot(data=plot_frame, x=x_col, y=metric, ax=ax, color="#2a6f97")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)
    return [_save_figure(fig, vis_dir / "metrics_overview.png", overwrite=overwrite)]


def _plot_runtime_breakdown(result_dir: Path, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    path = result_dir / "gpu_runtime.json"
    if not path.exists():
        return []
    runtime = json.loads(path.read_text())
    rows = []
    for key, value in runtime.items():
        if not isinstance(value, (int, float)):
            continue
        if "seconds" not in key:
            continue
        if float(value) <= 0:
            continue
        rows.append({"component": key, "seconds": float(value)})
    if not rows:
        return []
    frame = pd.DataFrame(rows).sort_values("seconds", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(frame) + 2)))
    ax.barh(frame["component"], frame["seconds"], color="#3d5a80")
    ax.set_title("Runtime Breakdown")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("")
    return [_save_figure(fig, vis_dir / "runtime_breakdown.png", overwrite=overwrite)]


def _plot_prediction_diagnostics(result_dir: Path, prediction_path: Path, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    frame = _safe_read_csv(prediction_path)
    if frame.empty:
        return []
    if {"y", "prob"}.issubset(frame.columns):
        return _plot_binary_predictions(frame, vis_dir, overwrite=overwrite)
    if {"AUC", "auc_hat"}.issubset(frame.columns):
        return _plot_regression_predictions(frame, vis_dir, overwrite=overwrite)
    if {"time", "event", "value_hat"}.issubset(frame.columns):
        return _plot_survival_predictions(frame, vis_dir, overwrite=overwrite)
    return []


def _plot_regression_predictions(frame: pd.DataFrame, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    frame = frame.copy()
    frame = frame[np.isfinite(frame["AUC"]) & np.isfinite(frame["auc_hat"])]
    if frame.empty:
        return []
    sample = _sample_frame(frame, 15000)
    residual = sample["auc_hat"] - sample["AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter = axes[0]
    if "split" in sample.columns and sample["split"].nunique() > 1:
        sns.scatterplot(data=sample, x="AUC", y="auc_hat", hue="split", s=14, alpha=0.55, ax=scatter)
    else:
        scatter.scatter(sample["AUC"], sample["auc_hat"], s=14, alpha=0.55, color="#2a6f97")
    limits = [
        float(np.nanmin([sample["AUC"].min(), sample["auc_hat"].min()])),
        float(np.nanmax([sample["AUC"].max(), sample["auc_hat"].max()])),
    ]
    scatter.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    scatter.set_title("Predicted vs True AUC")
    scatter.set_xlabel("True AUC")
    scatter.set_ylabel("Predicted AUC")

    hist = axes[1]
    hist.hist(residual, bins=40, color="#ee6c4d", alpha=0.85)
    hist.axvline(0.0, color="black", linestyle="--", linewidth=1)
    hist.set_title("Residual Distribution")
    hist.set_xlabel("auc_hat - AUC")
    hist.set_ylabel("Count")

    return [_save_figure(fig, vis_dir / "prediction_diagnostics.png", overwrite=overwrite)]


def _plot_binary_predictions(frame: pd.DataFrame, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    frame = frame.copy()
    frame = frame[np.isfinite(frame["y"]) & np.isfinite(frame["prob"])]
    if frame.empty:
        return []
    y = frame["y"].astype(int).to_numpy()
    prob = frame["prob"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    roc_score = roc_auc_score(y, prob)
    pr_score = average_precision_score(y, prob)

    axes[0, 0].plot(fpr, tpr, color="#2a6f97", linewidth=2, label=f"AUROC={roc_score:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend(loc="lower right")

    axes[0, 1].plot(recall, precision, color="#ee6c4d", linewidth=2, label=f"AUPRC={pr_score:.3f}")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend(loc="lower left")

    cal = _calibration_frame(frame)
    if not cal.empty:
        axes[1, 0].plot(cal["score_mean"], cal["response_rate"], marker="o", color="#3d5a80", linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[1, 0].set_title("Calibration by Quantile")
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 0].set_ylabel("Observed response rate")

    axes[1, 1].hist(frame["prob"], bins=30, color="#98c1d9", alpha=0.9)
    axes[1, 1].set_title("Predicted Probability Distribution")
    axes[1, 1].set_xlabel("prob")
    axes[1, 1].set_ylabel("Count")

    return [_save_figure(fig, vis_dir / "prediction_diagnostics.png", overwrite=overwrite)]


def _calibration_frame(frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    try:
        bins = pd.qcut(frame["prob"], q=min(n_bins, max(1, frame["prob"].nunique())), duplicates="drop")
    except Exception:
        return pd.DataFrame()
    grouped = frame.groupby(bins, observed=True)
    out = grouped.agg(score_mean=("prob", "mean"), response_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return out


def _plot_ablation_summary(result_dir: Path, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    frame = _safe_read_csv(result_dir / "ablation_summary.csv")
    if frame.empty or "variant" not in frame.columns:
        return []
    metric = None
    for candidate in ["within_drug_pcc_mean", "auroc", "overall_pcc", "value_spearman"]:
        if candidate in frame.columns:
            metric = candidate
            break
    if metric is None:
        numeric = [
            col
            for col in frame.columns
            if pd.api.types.is_numeric_dtype(frame[col]) and col not in {"n", "threshold", "n_pos", "n_neg"}
        ]
        if not numeric:
            return []
        metric = numeric[0]
    plot_frame = frame.copy()
    if "split" in plot_frame.columns:
        plot_frame = plot_frame.loc[plot_frame["split"].eq("test")].copy()
        if plot_frame.empty:
            plot_frame = frame.copy()
    plot_frame = plot_frame[["variant", metric]].dropna().sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(plot_frame) + 2)))
    sns.barplot(data=plot_frame, x=metric, y="variant", ax=ax, color="#2a9d8f")
    ax.set_title(f"Ablation Summary: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Variant")
    return [_save_figure(fig, vis_dir / "ablation_summary.png", overwrite=overwrite)]


def _plot_survival_predictions(frame: pd.DataFrame, vis_dir: Path, *, overwrite: bool = True) -> list[Path]:
    frame = frame.copy()
    frame = frame[np.isfinite(frame["time"]) & np.isfinite(frame["event"]) & np.isfinite(frame["value_hat"])]
    if frame.empty:
        return []
    sample = _sample_frame(frame, 15000)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if "event" in sample.columns and sample["event"].nunique() > 1:
        sns.scatterplot(data=sample, x="time", y="value_hat", hue="event", s=14, alpha=0.55, ax=axes[0])
    else:
        axes[0].scatter(sample["time"], sample["value_hat"], s=14, alpha=0.55, color="#2a6f97")
    axes[0].set_title("Value Prediction vs Survival Time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("value_hat")

    axes[1].hist(sample.loc[sample["event"].astype(int).eq(1), "value_hat"], bins=30, alpha=0.75, label="event=1", color="#ee6c4d")
    axes[1].hist(sample.loc[sample["event"].astype(int).eq(0), "value_hat"], bins=30, alpha=0.6, label="event=0", color="#3d5a80")
    axes[1].set_title("Value Distribution by Event")
    axes[1].set_xlabel("value_hat")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="best")
    return [_save_figure(fig, vis_dir / "survival_diagnostics.png", overwrite=overwrite)]


def _plot_latent_projection(
    result_dir: Path,
    prediction_path: Path | None,
    latent_path: Path,
    vis_dir: Path,
    *,
    overwrite: bool = True,
) -> list[Path]:
    if not latent_path.exists() or prediction_path is None or not prediction_path.exists():
        return []
    latents = np.load(latent_path, mmap_mode="r")
    pred = _safe_read_csv(prediction_path)
    if len(pred) == 0 or len(latents) == 0:
        return []
    n = min(len(pred), len(latents))
    if n < 2:
        return []
    pred = pred.iloc[:n].reset_index(drop=True)
    latents = np.asarray(latents[:n], dtype=np.float32)
    sample_idx = _sample_indices(n, 12000)
    pred_sample = pred.iloc[sample_idx].reset_index(drop=True)
    latent_sample = latents[sample_idx]

    outputs: list[Path] = []
    pca = PCA(n_components=2, random_state=0)
    pca_embed = pca.fit_transform(latent_sample)
    outputs.append(_save_figure(_scatter_projection(pca_embed, pred_sample, "PCA", color_kind=_latent_color_kind(pred_sample)), vis_dir / "latent_pca.png", overwrite=overwrite))
    return outputs


def _latent_color_kind(frame: pd.DataFrame) -> str:
    if {"y", "prob"}.issubset(frame.columns):
        return "binary"
    if {"AUC", "auc_hat"}.issubset(frame.columns):
        return "regression"
    return "split"


def _scatter_projection(embedding: np.ndarray, frame: pd.DataFrame, title: str, *, color_kind: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    x = embedding[:, 0]
    y = embedding[:, 1]
    if color_kind == "binary":
        values = frame["prob"].astype(float).to_numpy()
        sc = ax.scatter(x, y, c=values, cmap="viridis", s=12, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="prob")
    elif color_kind == "regression":
        values = np.abs(frame["auc_hat"].astype(float).to_numpy() - frame["AUC"].astype(float).to_numpy())
        sc = ax.scatter(x, y, c=values, cmap="magma", s=12, alpha=0.8)
        fig.colorbar(sc, ax=ax, label="|auc_hat - AUC|")
    elif "split" in frame.columns and frame["split"].nunique() > 1:
        sns.scatterplot(data=frame.assign(_x=x, _y=y), x="_x", y="_y", hue="split", s=12, alpha=0.8, ax=ax)
    else:
        ax.scatter(x, y, s=12, alpha=0.8, color="#2a6f97")
    ax.set_title(f"{title} Latent Projection")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return fig


def _sample_indices(n_rows: int, max_points: int) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def _sample_frame(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.iloc[_sample_indices(len(frame), max_points)].reset_index(drop=True)


def _ensure_terminal_latents(
    result_dir: Path,
    *,
    config,
    prediction_path: Path | None,
    device: str | None,
    overwrite: bool,
    summary: dict[str, Any],
) -> Path | None:
    latent_path = result_dir / "terminal_latents.npy"
    if latent_path.exists() and not overwrite:
        return latent_path
    if latent_path.exists():
        return latent_path

    try:
        if getattr(config, "task_type", None) == "tcga_binary_response" or (result_dir / "tcga_binary_scores.csv").exists():
            return _recompute_tcga_binary_latents(result_dir, latent_path, device=device, summary=summary)
        if (result_dir / "tcga_actual_treatment_scores.csv").exists():
            return _recompute_tcga_actual_treatment_latents(result_dir, latent_path, device=device, summary=summary)
        if prediction_path is not None and (result_dir / "model.pt").exists() and (result_dir / "artifacts.pkl").exists():
            return _recompute_gdsc_latents(result_dir, latent_path, config=config, device=device, summary=summary)
    except Exception as exc:  # pragma: no cover - keep visual backfill best-effort
        summary["skipped"].append(f"latent_export_failed:{type(exc).__name__}")
        summary["errors"].append({"step": "latent_materialization", "error": f"{type(exc).__name__}: {exc}"})
        return None
    summary["skipped"].append("latent_export_unavailable")
    return None


def _recompute_gdsc_latents(
    result_dir: Path,
    latent_path: Path,
    *,
    config,
    device: str | None,
    summary: dict[str, Any],
) -> Path:
    from .train import apply_cell_train_statistics_feature_selection, load_model, load_prepared, predict_frame

    benchmark_dir = result_dir.parent.parent
    prepared_path = benchmark_dir / "data" / "processed" / "default" / "prepared.pkl"
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared.pkl not found for backfill: {prepared_path}")
    prepared = load_prepared(prepared_path)
    if config is not None:
        prepared = apply_cell_train_statistics_feature_selection(prepared, config)
    model = load_model(result_dir, prepared.artifacts, strict=False)
    if config is None:
        config = load_benchmark_config(result_dir / "default.yaml")
    predict_frame(
        model,
        prepared.responses,
        prepared,
        batch_size=4096,
        device=device or "cpu",
        benchmark=config,
        controls=getattr(config, "controls", None),
        terminal_latent_path=latent_path,
    )
    summary["generated"].append(str(latent_path))
    return latent_path


def _recompute_tcga_binary_latents(result_dir: Path, latent_path: Path, *, device: str | None, summary: dict[str, Any]) -> Path:
    from .tcga_binary import load_model, load_prepared, predict_tcga_binary

    benchmark_dir = result_dir.parent.parent
    prepared_path = benchmark_dir / "data" / "processed" / "default" / "prepared.pkl"
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared.pkl not found for backfill: {prepared_path}")
    prepared = load_prepared(prepared_path)
    model = load_model(result_dir, prepared.artifacts, strict=False)
    predict_tcga_binary(
        model,
        prepared,
        device=device or "cpu",
        batch_size=4096,
        terminal_latent_path=latent_path,
    )
    summary["generated"].append(str(latent_path))
    return latent_path


def _recompute_tcga_actual_treatment_latents(result_dir: Path, latent_path: Path, *, device: str | None, summary: dict[str, Any]) -> Path:
    import pickle

    from .config import Paths
    from .external import evaluate_tcga_actual_treatments
    from .train import load_model

    with (result_dir / "artifacts.pkl").open("rb") as f:
        artifacts = pickle.load(f)
    model = load_model(result_dir, artifacts, strict=False)
    evaluate_tcga_actual_treatments(
        model,
        artifacts,
        Paths(),
        result_dir,
        device=device or "cpu",
        export_terminal_latents=True,
        terminal_latent_path=latent_path,
    )
    summary["generated"].append(str(latent_path))
    return latent_path


def _write_index_files(result_dir: Path, vis_dir: Path, summary: dict[str, Any]) -> None:
    files = sorted({Path(path).name for path in summary["generated"] if Path(path).parent == vis_dir})
    index_lines = [
        f"# Visualizations for {result_dir.name}",
        "",
        f"- Result directory: `{result_dir}`",
        f"- Generated files: {len(files)}",
        "",
    ]
    for file_name in files:
        index_lines.append(f"- [{file_name}](./{file_name})")
    (vis_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
    write_json(vis_dir / "summary.json", summary)
