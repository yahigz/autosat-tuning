"""Plotting helpers for FCNS tuning runs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


@dataclass(frozen=True)
class PlotScale:
    time_max: float
    color_max: float
    iteration_max: int


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name))


def _best_series(trace: Sequence[Mapping[str, object]], fallback_color: int | None = None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    cur: float | None = None
    for item in sorted(trace, key=lambda p: float(p.get("elapsed", 0.0) or 0.0)):
        elapsed = float(item.get("elapsed", 0.0) or 0.0)
        colors = item.get("colors")
        if colors is None:
            continue
        cur = float(colors) if cur is None else min(cur, float(colors))
        points.append((elapsed, cur))
    if not points and fallback_color is not None:
        points.append((0.0, float(fallback_color)))
    return points


def _aggregate_series(runs: Sequence[Mapping[str, object]]) -> list[tuple[float, float]]:
    per_run = [
        _best_series(run.get("trace", []), fallback_color=int(run.get("final_colors", 0) or 0))
        for run in runs
    ]
    times = {0.0}
    for series in per_run:
        times.update(t for t, _ in series)
    ordered_times = sorted(times)
    result: list[tuple[float, float]] = []
    for t in ordered_times:
        total = 0.0
        for series in per_run:
            if not series:
                continue
            value = series[0][1]
            for point_time, point_value in series:
                if point_time <= t:
                    value = point_value
                else:
                    break
            total += value
        result.append((t, total))
    return result


def _style_axes(ax, scale: PlotScale, title: str, xlabel: str = "Time (s)", ylabel: str = "Colors"):
    ax.set_xlim(0, max(scale.time_max, 1e-9))
    ax.set_ylim(0, max(scale.color_max, 1.0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def save_iteration_plots(
    results_root: str | Path,
    iteration_index: int,
    runs: Sequence[Mapping[str, object]],
    scale: PlotScale,
    *,
    run_label: str = "",
) -> dict[str, Path]:
    if not _MATPLOTLIB_AVAILABLE:
        print("[Plotting] matplotlib unavailable; skipping plots.", flush=True)
        return {}

    results_root = Path(results_root)
    out_dir = results_root / "plots" / f"iter_{iteration_index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for run in runs:
        name = str(run.get("name", "instance"))
        trace = run.get("trace", [])
        best_points = _best_series(trace, fallback_color=int(run.get("final_colors", 0) or 0))
        if not best_points:
            continue

        xs = [p[0] for p in best_points]
        ys = [p[1] for p in best_points]
        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(xs, ys, c=xs, cmap="viridis", vmin=0, vmax=max(scale.time_max, 1e-9), s=28, alpha=0.9)
        ax.plot(xs, ys, color="black", linewidth=1.0, alpha=0.4)
        _style_axes(
            ax,
            scale,
            title=f"FCNS Iteration {iteration_index}: {name}" + (f"\n{run_label}" if run_label else ""),
        )
        fig.colorbar(scatter, ax=ax, label="Time used to reach point (s)")
        fig.tight_layout()
        path = out_dir / f"{_safe_name(name)}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written[name] = path

    aggregate = _aggregate_series(runs)
    if aggregate:
        xs = [p[0] for p in aggregate]
        ys = [p[1] for p in aggregate]
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(xs, ys, c=xs, cmap="viridis", vmin=0, vmax=max(scale.time_max, 1e-9), s=30, alpha=0.92)
        ax.plot(xs, ys, color="#1f2937", linewidth=1.2, alpha=0.75)
        _style_axes(
            ax,
            scale,
            title=f"FCNS Iteration {iteration_index}: aggregate colors" + (f"\n{run_label}" if run_label else ""),
            ylabel="Sum of colors",
        )
        fig.colorbar(scatter, ax=ax, label="Time used to reach point (s)")
        fig.tight_layout()
        path = out_dir / "aggregate.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written["aggregate"] = path

    return written


def save_overall_3d_plot(
    results_root: str | Path,
    all_runs_by_iteration: Sequence[Sequence[Mapping[str, object]]],
    scale: PlotScale,
    *,
    run_label: str = "",
) -> Path | None:
    if not _MATPLOTLIB_AVAILABLE:
        return None

    results_root = Path(results_root)
    out_dir = results_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[int] = []
    for iteration_index, runs in enumerate(all_runs_by_iteration):
        aggregate = _aggregate_series(runs)
        for elapsed, colors in aggregate:
            xs.append(elapsed)
            ys.append(colors)
            zs.append(iteration_index)

    if not xs:
        return None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=xs, cmap="viridis", s=18, alpha=0.85)
    ax.set_xlim(0, max(scale.time_max, 1e-9))
    ax.set_ylim(0, max(scale.color_max, 1.0))
    ax.set_zlim(0, max(scale.iteration_max, 1))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sum of colors")
    ax.set_zlabel("Iteration")
    ax.set_title("FCNS aggregate progress over iterations" + (f"\n{run_label}" if run_label else ""))
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="Time used to reach point (s)")
    fig.tight_layout()
    path = out_dir / "aggregate_3d.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
