"""Plotting utilities for solver tuning runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def _load_snapshots(snapshots_dir: Path):
    pairs = []
    if not snapshots_dir.exists():
        return pairs
    for sf in sorted(snapshots_dir.glob("iter_*_best.json")):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            it = int(data.get("iter", sf.stem.split("_")[1]))
            p = data.get("PAR-2")
            if p is not None:
                pairs.append((it, float(p)))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def plot_training_curve(
    results_root: str | Path,
    baseline_par2: Optional[float] = None,
    run_id: str = "",
    show: bool = False,
) -> Optional[Path]:
    if not _MATPLOTLIB_AVAILABLE:
        print("[Plotting] matplotlib not available - skipping plot generation.", flush=True)
        return None

    results_root = Path(results_root)
    snapshots_dir = results_root / "snapshots"
    pairs = _load_snapshots(snapshots_dir)

    if baseline_par2 is None:
        final_path = results_root / "final_result.json"
        if final_path.exists():
            try:
                final = json.loads(final_path.read_text(encoding="utf-8"))
                baseline_par2 = final.get("0", {}).get("PAR-2")
            except Exception:
                pass

    if not pairs and baseline_par2 is None:
        print("[Plotting] No data to plot.", flush=True)
        return None

    iters = [p[0] for p in pairs]
    par2s = [p[1] for p in pairs]

    if baseline_par2 is not None:
        iters = [0] + iters
        par2s = [float(baseline_par2)] + par2s

    best_so_far = []
    cur_best = float("inf")
    for value in par2s:
        cur_best = min(cur_best, value)
        best_so_far.append(cur_best)

    out_dir = results_root / "analysis_par2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "par2_vs_iter.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, par2s, marker="o", linewidth=1.2, alpha=0.7, label="PAR-2 (each iter)")
    ax.plot(iters, best_so_far, linestyle="--", linewidth=1.8, color="red", label="Best so far")
    if baseline_par2 is not None:
        ax.axhline(y=float(baseline_par2), color="gray", linestyle=":", linewidth=1.2, label="Baseline")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PAR-2 (seconds)")
    title = "Solver Tuning - PAR-2 vs Iteration"
    if run_id:
        title += f"\n{run_id}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[Plotting] Saved training curve -> {out_path}", flush=True)
    return out_path


def plot_all_runs(runs_root: str | Path = "./results/runs") -> int:
    runs_root = Path(runs_root)
    count = 0
    if not runs_root.exists():
        return count
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        result = plot_training_curve(results_root=entry, run_id=entry.name)
        if result is not None:
            count += 1
    return count
