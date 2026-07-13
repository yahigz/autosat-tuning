from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from autosat_core.marker_adapter import MarkerSolverAdapter
from autosat_core.server import start_server, write_progress
from autosat_core.tasks import TaskSpec, task_specs_from_config
from llm_api.base_api import get_llm_api

from prompting import (
    build_prompt_text_for_tasks,
    build_structured_schema_all,
    build_tool_schema_all,
    load_prompt_text,
    parse_multi_response,
    parse_structured_response,
)
from plotting import PlotScale, save_iteration_plots, save_overall_3d_plot


BASELINE_CPP = Path(__file__).resolve().parent / "solver" / "baseline" / "fcns.cpp"
TEMPLATE_CPP = Path(__file__).resolve().parent / "solver" / "template" / "fcns.cpp"
DEFAULT_PROMPT_ORIGINAL = Path(__file__).resolve().parent / "prompts" / "original_prompt.txt"
DEFAULT_PROMPT_FEEDBACK = Path(__file__).resolve().parent / "prompts" / "feedback_prompt.txt"
DEFAULT_TRAIN_DIR = Path(__file__).resolve().parent / "data" / "train"
DEFAULT_EVAL_DIR = Path(__file__).resolve().parent / "data" / "eval"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "runs"
DEFAULT_TEMP_ROOT = Path(__file__).resolve().parent / "temp" / "runs"


@dataclass(frozen=True)
class GraphInstance:
    name: str
    path: Path
    n: int
    edges: list[tuple[int, int]]
    stdin_text: str


@dataclass
class InstanceRun:
    name: str
    final_colors: int
    final_time: float
    greedy_valid: bool
    par2: float
    timed_out: bool
    trace: list[dict[str, Any]]
    stdout: str
    stderr: list[str]
    exit_code: int


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _log(message: str) -> None:
    print(f"[FCNS] {time.strftime('%H:%M:%S')} {message}", flush=True)


def _make_run_id(explicit: str = "") -> str:
    explicit = str(explicit or "").strip()
    return explicit or (time.strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload or {})


def _normalize_task_name(value: str) -> str:
    return str(value or "").strip().strip("/")


def _discover_tasks(config_payload: Mapping[str, Any]) -> tuple[MarkerSolverAdapter, list[TaskSpec], list[str]]:
    adapter = MarkerSolverAdapter(name="FCNS", baseline_cpp=BASELINE_CPP, template_cpp=TEMPLATE_CPP)
    available = adapter.available_task_names()
    if not available:
        raise ValueError(
            f"No marker-based tasks were found in {BASELINE_CPP}. Add marker pairs before running the pipeline."
        )

    baseline_text = adapter.baseline_text()
    baseline_codes = {task: adapter.extract_baseline_section(task, baseline_text=baseline_text) for task in available}
    task_specs = task_specs_from_config(
        config_payload,
        allowed_names=available,
        baseline_codes=baseline_codes,
        fallback_names=available,
    )
    if not task_specs:
        task_specs = [TaskSpec(name=name, baseline_code=baseline_codes.get(name, "")) for name in available]
    task_names = [task.name for task in task_specs]
    return adapter, task_specs, task_names


def _task_for_iteration(task_names: Sequence[str], selection_mode: str, iter_idx: int, rand_seed: int = 42) -> str:
    if not task_names:
        raise ValueError("task_names must not be empty")
    mode = str(selection_mode or "random_one").strip().lower()
    if mode == "all":
        return "__all__"
    if mode == "cycle":
        return task_names[iter_idx % len(task_names)]
    if mode == "random_one":
        import random

        return random.Random(int(rand_seed)).choice(list(task_names))
    raise ValueError(f"Unsupported task_selection_mode: {selection_mode!r}")


def _iter_graph_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = [p for p in sorted(data_dir.iterdir()) if p.is_file()]
    if not files:
        raise ValueError(f"No graph files found in {data_dir}")
    return files


def _decode_binary_upper_triangle(payload: bytes, n: int) -> list[tuple[int, int]]:
    possible = n * (n - 1) // 2
    prefix = [0]
    running = 0
    for row in range(n):
        running += n - row - 1
        prefix.append(running)

    def _pair_from_index(index: int) -> tuple[int, int]:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix[mid] <= index < prefix[mid + 1]:
                lo = mid
                break
            if index < prefix[mid]:
                hi = mid
            else:
                lo = mid + 1
        row = lo
        col = row + 1 + (index - prefix[row])
        return row, col

    edges: list[tuple[int, int]] = []
    bit_index = 0
    for byte in payload:
        for bit in range(8):
            if bit_index >= possible:
                break
            if byte & (1 << bit):
                edges.append(_pair_from_index(bit_index))
            bit_index += 1
        if bit_index >= possible:
            break
    return edges


def _parse_graph_file(path: Path) -> GraphInstance:
    raw = path.read_bytes()
    p_match = re.search(rb"^p\s+edge\s+(\d+)\s+(\d+)", raw, re.MULTILINE)
    if p_match is not None:
        n = int(p_match.group(1))
        header_end = raw.find(b"\n", p_match.end())
        if header_end == -1:
            header_end = len(raw)
        payload = raw[header_end + 1 :]

        if path.suffix == ".b" or any(byte == 0 or byte > 0x7F for byte in payload):
            edges = _decode_binary_upper_triangle(payload, n)
        else:
            text_payload = payload.decode("latin1", errors="ignore")
            numbers = [int(value) for value in re.findall(r"-?\d+", text_payload)]
            if len(numbers) % 2 != 0:
                raise ValueError(f"Odd number of vertex ids in {path}")
            edges = list(zip(numbers[0::2], numbers[1::2]))
    else:
        text = raw.decode("latin1", errors="ignore")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Could not parse graph file: {path}")
        first_numbers = [int(value) for value in re.findall(r"-?\d+", lines[0])]
        if len(first_numbers) < 2:
            raise ValueError(f"Could not read graph header from {path}")
        n = int(first_numbers[0])
        payload_numbers = [int(value) for value in re.findall(r"-?\d+", "\n".join(lines[1:]))]
        if len(payload_numbers) % 2 != 0:
            raise ValueError(f"Odd number of vertex ids in {path}")
        edges = list(zip(payload_numbers[0::2], payload_numbers[1::2]))

    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if a < 0 or b < 0 or a >= n or b >= n:
            continue
        if (a, b) not in seen:
            seen.add((a, b))
            normalized.append((a, b))

    stdin_text = [f"{n} {len(normalized)}"]
    stdin_text.extend(f"{u} {v}" for u, v in normalized)
    return GraphInstance(
        name=path.stem,
        path=path,
        n=n,
        edges=normalized,
        stdin_text="\n".join(stdin_text) + "\n",
    )


def _load_instances(data_dir: Path) -> list[GraphInstance]:
    return [_parse_graph_file(path) for path in _iter_graph_files(data_dir)]


def _stderr_metrics(line: str) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        return {"raw": line.rstrip("\n")}

    data: dict[str, Any] = {}
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                data.update(parsed)
        except Exception:
            pass

    if not data:
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", stripped):
            key = key.lower()
            if "." in value:
                data[key] = float(value)
            else:
                data[key] = int(value)

    if "colors" not in data:
        for key in ("colors", "color", "best_colors", "chromatic", "chromatic_number", "k"):
            if key in data:
                data["colors"] = int(data[key])
                break

    if "elapsed" not in data:
        for key in ("elapsed", "time", "seconds", "t"):
            if key in data:
                data["elapsed"] = float(data[key])
                break

    data["raw"] = line.rstrip("\n")
    return data


def _compile_solver(source_cpp: Path, executable_path: Path) -> bool:
    executable_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O3", "-Wall", "-Wextra", "-std=c++17", str(source_cpp), "-o", str(executable_path)]
    _log(f"Compiling {source_cpp.name}")
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
    return result.returncode == 0


def _run_solver(executable_path: Path, instance: GraphInstance, timeout_s: float) -> InstanceRun:
    proc = subprocess.Popen(
        [str(executable_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stdout_chunks: list[str] = []
    stderr_lines: list[str] = []
    trace: list[dict[str, Any]] = []
    start_perf = time.perf_counter()
    start_epoch = time.time()

    def _read_stdout() -> None:
        assert proc.stdout is not None
        for chunk in proc.stdout:
            stdout_chunks.append(chunk)

    def _read_stderr() -> None:
        assert proc.stderr is not None
        for chunk in proc.stderr:
            stderr_lines.append(chunk)
            metrics = _stderr_metrics(chunk)
            metrics["elapsed"] = float(metrics.get("elapsed", time.perf_counter() - start_perf))
            metrics["captured_at"] = start_epoch + float(metrics["elapsed"])
            trace.append(metrics)

    def _feed_stdin() -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(instance.stdin_text)
            proc.stdin.close()
        except Exception:
            try:
                proc.stdin.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdin_thread = threading.Thread(target=_feed_stdin, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait(timeout=5)
    finally:
        stdin_thread.join(timeout=1)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

    stdout_text = "".join(stdout_chunks)
    stdout_lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    final_colors = instance.n
    final_time = time.perf_counter() - start_perf
    greedy_valid = True
    if stdout_lines:
        try:
            final_colors = int(stdout_lines[0].split()[0])
        except Exception:
            pass
    if trace:
        color_points = [float(item["colors"]) for item in trace if isinstance(item.get("colors"), (int, float))]
        time_points = [float(item["elapsed"]) for item in trace if isinstance(item.get("elapsed"), (int, float))]
        greedy_valid = any(item.get("greedy_valid") is False for item in trace)
        if color_points:
            final_colors = int(color_points[-1])
        if time_points:
            final_time = max(time_points)

    return InstanceRun(
        name=instance.name,
        final_colors=final_colors,
        final_time=final_time,
        greedy_valid=greedy_valid,
        trace=trace,
        stdout=stdout_text,
        stderr=stderr_lines,
        exit_code=proc.returncode or 0,
        timed_out=timed_out,
        par2=2 * timeout_s if timed_out else final_time
    )


def _evaluate_source(
    source_cpp: Path,
    executable_path: Path,
    instances: Sequence[GraphInstance],
    timeout_s: float,
    max_workers: int = 1,
) -> tuple[dict[str, Any], list[InstanceRun]]:
    if not _compile_solver(source_cpp, executable_path):
        return {"compile_ok": False, "score": [10**9, float("inf")]}, []

    worker_count = max(1, int(max_workers or 1))
    _log(f"Evaluating {len(instances)} instances with {worker_count} worker(s)")
    if worker_count > 1 and len(instances) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_count, len(instances))) as executor:
            runs = list(executor.map(lambda instance: _run_solver(executable_path, instance, timeout_s), instances))
    else:
        runs = [_run_solver(executable_path, instance, timeout_s) for instance in instances]

    total_colors = 0
    total_time = 0.0
    all_ok = True
    timed_out = 0
    par2 = 0.0
    for run in runs:
        total_colors += int(run.final_colors)
        total_time += float(run.final_time)
        all_ok = all_ok and (not run.timed_out) and run.exit_code == 0
        timed_out += 1 if run.timed_out and run.exit_code == 0 else 0
        par2 += 2 * timeout_s if run.timed_out else float(run.final_time)

    return (
        {
            "compile_ok": True,
            "score": [total_colors, par2],
            "all_ok": all_ok,
            "total_colors": total_colors,
            "total_time": total_time,
            "timed_out": timed_out,
            "par2": par2,
            "greedy_valid": all(run.greedy_valid for run in runs),
        },
        runs,
    )


def _build_prompt(
    prompt_path: Path,
    task_specs: Sequence[TaskSpec],
    baseline_codes: Mapping[str, str],
    *,
    baseline_info: Mapping[str, Any] | None,
    last_iter_result: Mapping[str, Any] | None,
    structured_output: bool,
    all_tasks_mode: bool,
    exploration_stats_section: str = "",
    best_results_section: str = "",
) -> str:
    base_prompt_text = load_prompt_text(str(prompt_path))
    return build_prompt_text_for_tasks(
        base_prompt_text=base_prompt_text,
        task_specs=task_specs,
        baseline_info=baseline_info,
        baseline_codes=baseline_codes,
        last_iter_result=last_iter_result,
        structured_output=structured_output,
        all_tasks_mode=all_tasks_mode,
        exploration_stats_section=exploration_stats_section,
        best_results_section=best_results_section,
    )


def _query_llm(prompt_file: Path, args: argparse.Namespace, task_name: str) -> dict[str, str]:
    llm_api = get_llm_api(args)
    temperature = float(getattr(args, "temperature", 1.0) or 1.0)
    if getattr(llm_api, "_structured_output", False):
        return llm_api.call_api_structured(prompt_file=str(prompt_file), temperature=temperature, task_name=task_name)
    raw = llm_api.call_api(prompt_file=str(prompt_file), temperature=temperature)
    parsed = parse_structured_response(raw)
    return parsed


def _query_llm_all(
    prompt_file: Path,
    args: argparse.Namespace,
    task_names: Sequence[str],
    enable_exploration: bool = False,
) -> dict[str, dict[str, str]]:
    llm_api = get_llm_api(args)
    temperature = float(getattr(args, "temperature", 1.0) or 1.0)
    prompt_text = prompt_file.read_text(encoding="utf-8")

    if getattr(llm_api, "_structured_output", False):
        if llm_api.__class__.__name__ == "GPTCallAPI":
            raw = llm_api.call_structured_all(prompt_text, temperature, build_structured_schema_all(task_names, enable_exploration=enable_exploration))
            return parse_multi_response(raw, task_names)
        if llm_api.__class__.__name__ == "ElizaCallAPI":
            raw = llm_api.call_structured_all(prompt_text, temperature, build_tool_schema_all(task_names, enable_exploration=enable_exploration))
            return parse_multi_response(raw, task_names)

    raw = llm_api.call_api(prompt_file=str(prompt_file), temperature=temperature)
    return parse_multi_response(raw, task_names)


def _select_better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_key = (left.get("greedy_valid", True), left.get("timed_out", 0), float(left.get("total_colors", float("inf"))), float(left.get("total_time", float("inf"))))
    right_key = (right.get("greedy_valid", True), right.get("timed_out", 0), float(right.get("total_colors", float("inf"))), float(right.get("total_time", float("inf"))))
    return right_key < left_key


def _baseline_result(
    adapter: MarkerSolverAdapter,
    task_specs: Sequence[TaskSpec],
    baseline_codes: Mapping[str, str],
    prompt_path: Path,
    args: argparse.Namespace,
    results_root: Path,
    temp_root: Path,
    instances: Sequence[GraphInstance],
    scale: PlotScale,
    max_workers: int,
    source_text: str | None = None,
) -> tuple[dict[str, Any], list[InstanceRun], dict[str, str]]:
    _log("Running baseline evaluation")
    baseline_rendered = adapter.render_source({}, baseline_text=source_text)
    baseline_cpp = temp_root / "baseline" / "fcns.cpp"
    baseline_cpp.parent.mkdir(parents=True, exist_ok=True)
    baseline_cpp.write_text(baseline_rendered, encoding="utf-8")
    executable = temp_root / "baseline" / "fcns"
    summary, runs = _evaluate_source(baseline_cpp, executable, instances, float(args.timeout), max_workers=max_workers)
    best_source = baseline_rendered
    _save_candidate_artifacts(results_root, 0, 0, summary, runs, best_source)
    save_iteration_plots(results_root, 0, [
        {
            "name": run.name,
            "final_colors": run.final_colors,
            "final_time": run.final_time,
            "trace": run.trace,
            "timed_out": run.timed_out,
            "par2": run.par2,
            "greedy_valid": run.greedy_valid,
        }
        for run in runs
    ], scale, run_label=str(getattr(args, "run_id", "")))
    _atomic_write_json(results_root / "baseline_result.json", summary)
    return summary, runs, {"source": best_source}


def _progress_payload(
    run_id: str,
    baseline_summary: Mapping[str, Any],
    best_state: Mapping[str, Any],
    iterations_log: Sequence[Mapping[str, Any]],
    primary_label: str = "colors",
    secondary_label: str = "par2",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "metric_mode": f"{primary_label}_{secondary_label}",
        "metric_labels": {"primary": primary_label, "secondary": secondary_label},
        "baseline": baseline_summary,
        "best": best_state,
        "iterations": list(iterations_log),
    }


def _save_candidate_artifacts(
    results_root: Path,
    iteration_index: int,
    candidate_index: int,
    summary: Mapping[str, Any],
    runs: Sequence[InstanceRun],
    rendered_source: str,
) -> None:
    iter_dir = results_root / "iterations" / f"iter_{iteration_index:04d}" / f"candidate_{candidate_index:02d}"
    traces_dir = iter_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(iter_dir / "summary.json", dict(summary))
    (iter_dir / "solver.cpp").write_text(rendered_source, encoding="utf-8")
    for run in runs:
        _atomic_write_json(
            traces_dir / f"{_normalize_task_name(run.name)}.json",
            {
                "name": run.name,
                "final_colors": run.final_colors,
                "final_time": run.final_time,
                "trace": run.trace,
                "stdout": run.stdout,
                "stderr": run.stderr,
                "exit_code": run.exit_code,
                "timed_out": run.timed_out,
                "par2": run.par2,
                "greedy_valid": run.greedy_valid,
            },
        )


def _render_candidate_source(
    adapter: MarkerSolverAdapter,
    task_names: Sequence[str],
    payload: Mapping[str, Any],
    all_tasks_mode: bool,
) -> dict[str, str]:
    if all_tasks_mode:
        return {task: str(payload.get(task, {}).get("code", "") or "") for task in task_names}
    return {str(task_names[0]): str(payload.get("code", "") or "")}


def _iter_prompt_context(
    task_names: Sequence[str],
    payload: Mapping[str, Any],
    rendered_source: str,
    all_tasks_mode: bool,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if all_tasks_mode:
        implementations: list[dict[str, Any]] = []
        for task in task_names:
            item = payload.get(task, {}) if isinstance(payload.get(task, {}), Mapping) else {}
            implementations.append(
                {
                    "task": task,
                    "code": str(item.get("code", "") or ""),
                    "title": str(item.get("title", "") or ""),
                    "reason": str(item.get("reason", "") or ""),
                }
            )
        return {"implementations": implementations, "colors": summary.get("total_colors") if summary else None, "par2": summary.get("par2") if summary else None, "timed_out": summary.get("timed_out") if summary else None, "time": summary.get("total_time") if summary else None, "greedy_valid": summary.get("greedy_valid") if summary else None}
    
    return {
        "code": str(payload.get("code", "") or ""),
        "title": str(payload.get("title", "") or ""),
        "reason": str(payload.get("reason", "") or ""),
        "colors": summary.get("total_colors") if summary else None,
        "par2": summary.get("par2") if summary else None,
        "time": summary.get("total_time") if summary else None,
        "timed_out": summary.get("timed_out") if summary else None,
        "greedy_valid": summary.get("greedy_valid") if summary else None,
    }


def _write_progress(results_root: Path, payload: Mapping[str, Any]) -> None:
    write_progress(payload, progress_file=str(results_root / "progress.json"))


def _format_best_results_section(
    best_by_primary: Mapping[str, Any],
    best_by_secondary: Mapping[str, Any],
    primary_label: str,
    secondary_label: str,
    primary_objective: str,
) -> str:
    direction = "lower" if primary_objective == "min" else "higher"
    lines = [
        "=== Best results (tracked across all iterations) ===",
        f"PRIMARY metric: {primary_label} ({direction} is better)",
    ]
    bp = best_by_primary
    bs = best_by_secondary
    lines.append(
        f"Best by {primary_label:<10}: iter={bp.get('iter', '?'):<4} "
        f"colors={bp.get('total_colors', '?'):<6} "
        f"par2={bp.get('par2', '?'):.2f}  "
        f"time={bp.get('total_time', '?'):.2f}s"
    )
    lines.append(
        f"Best by {secondary_label:<10}: iter={bs.get('iter', '?'):<4} "
        f"colors={bs.get('total_colors', '?'):<6} "
        f"par2={bs.get('par2', '?'):.2f}  "
        f"time={bs.get('total_time', '?'):.2f}s"
    )
    return "\n".join(lines)


def _execute_exploration_code(
    code: str,
    instances: Sequence[Any],
    timeout_per_instance: float,
) -> dict[str, dict[str, float]]:
    """Run model-generated get_statistics(n, m, adj_list) on all instances and aggregate."""
    if not code or not code.strip():
        return {}

    # Strip markdown code fences the model may include
    stripped = re.sub(r"^```[a-zA-Z]*\n?", "", code.strip())
    stripped = re.sub(r"\n?```$", "", stripped)
    code = stripped.strip()
    if not code:
        return {}

    local_ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<exploration_code>", "exec"), {}, local_ns)  # noqa: S102
    except Exception as exc:
        _log(f"[Exploration] exec error: {exc}")
        return {}

    fn = local_ns.get("get_statistics")
    if not callable(fn):
        _log("[Exploration] get_statistics not found in exploration_code")
        return {}

    per_instance_results: list[dict[str, float]] = []
    for inst in instances:
        adj_list = [[] for _ in range(inst.n)]
        for u, v in inst.edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        result_holder: list[Any] = [None]
        error_holder: list[Any] = [None]

        def _run(fn=fn, n=inst.n, m=len(inst.edges), adj=adj_list, rh=result_holder, eh=error_holder):
            try:
                rh[0] = fn(n, m, adj)
            except Exception as exc:
                eh[0] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_per_instance)
        if t.is_alive():
            _log(f"[Exploration] timeout on instance {inst.name}")
            continue
        if error_holder[0] is not None:
            _log(f"[Exploration] error on {inst.name}: {error_holder[0]}")
            continue
        raw = result_holder[0]
        if not isinstance(raw, dict):
            _log(f"[Exploration] non-dict result on {inst.name}")
            continue
        validated = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if validated:
            per_instance_results.append(validated)

    if not per_instance_results:
        return {}

    all_keys: set[str] = set()
    for r in per_instance_results:
        all_keys.update(r.keys())

    aggregated: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [r[key] for r in per_instance_results if key in r]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        aggregated[key] = {
            "mean": round(mean, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(variance ** 0.5, 4),
        }
    return aggregated


def _format_exploration_stats_section(accumulated: Mapping[str, Any]) -> str:
    if not accumulated:
        return ""
    already = ", ".join(sorted(accumulated.keys()))
    lines = [
        "=== Exploration Statistics (computed on training data) ===",
        f"Already collected — do NOT re-collect these keys: {already}",
    ]
    for key in sorted(accumulated.keys()):
        agg = accumulated[key]
        lines.append(
            f"  {key:<20}: mean={agg.get('mean', '?'):<8} "
            f"min={agg.get('min', '?'):<8} "
            f"max={agg.get('max', '?'):<8} "
            f"std={agg.get('std', '?')}"
        )
    lines.append("Provide NEW stat names in exploration_code that are not listed above.")
    return "\n".join(lines)


def _load_exploration_stats(results_root: Path) -> dict[str, Any]:
    path = results_root / "exploration_stats.json"
    if not path.exists():
        return {"accumulated": {}, "iterations": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"accumulated": {}, "iterations": []}


def _load_checkpoint(checkpoint_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_dir / "latest_checkpoint.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _save_checkpoint(checkpoint_dir: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(checkpoint_dir / "latest_checkpoint.json", dict(payload))


def _evaluate_best_on_eval(
    adapter: MarkerSolverAdapter,
    rendered_source: str,
    eval_instances: Sequence[GraphInstance],
    results_root: Path,
    temp_root: Path,
    timeout_s: float,
    max_workers: int = 1,
) -> dict[str, Any]:
    eval_cpp = temp_root / "eval" / "fcns.cpp"
    eval_cpp.parent.mkdir(parents=True, exist_ok=True)
    eval_cpp.write_text(rendered_source, encoding="utf-8")
    executable = temp_root / "eval" / "fcns"
    summary, runs = _evaluate_source(eval_cpp, executable, eval_instances, timeout_s, max_workers=max_workers)
    _atomic_write_json(results_root / "eval_best_result.json", summary)
    _save_candidate_artifacts(results_root, -1, 0, summary, runs, rendered_source)
    return summary


def _graceful_shutdown(signum: int, frame) -> None:
    raise KeyboardInterrupt(f"Signal {signum}")


def main(args: argparse.Namespace) -> None:
    config_payload = dict(getattr(args, "_loaded_config_payload", {}) or {})
    adapter, task_specs, task_names = _discover_tasks(config_payload)

    run_id = _make_run_id(getattr(args, "run_id", ""))
    results_root = Path(getattr(args, "results_root", DEFAULT_RESULTS_ROOT)) / run_id
    temp_root = Path(getattr(args, "temp_root", DEFAULT_TEMP_ROOT)) / run_id
    results_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    prompt_original = Path(config_payload.get("prompt_original", DEFAULT_PROMPT_ORIGINAL))
    prompt_feedback = Path(config_payload.get("prompt_feedback", DEFAULT_PROMPT_FEEDBACK))
    train_data_dir = Path(config_payload.get("train_data_dir", DEFAULT_TRAIN_DIR))
    eval_data_dir = Path(config_payload.get("eval_data_dir", DEFAULT_EVAL_DIR))
    baseline_cpp = Path(config_payload.get("baseline_cpp", BASELINE_CPP))
    template_cpp = Path(config_payload.get("template_cpp", TEMPLATE_CPP))
    timeout_s = float(config_payload.get("timeout", getattr(args, "timeout", 10)) or 10)
    data_parallel_size = int(config_payload.get("data_parallel_size", getattr(args, "data_parallel_size", 1)) or 1)
    eval_parallel_size = int(config_payload.get("eval_parallel_size", getattr(args, "eval_parallel_size", 1)) or 1)
    iteration_num = int(config_payload.get("iteration_num", getattr(args, "iteration_num", 10)) or 10)
    batch_size = int(config_payload.get("batch_size", getattr(args, "batch_size", 1)) or 1)
    selection_mode = str(config_payload.get("task_selection_mode", getattr(args, "task_selection_mode", "all")) or "all")
    rand_seed = int(config_payload.get("rand_seed", getattr(args, "rand_seed", 42)) or 42)
    structured_output = bool(config_payload.get("use_structured", getattr(args, "use_structured", True)))

    args.run_id = run_id
    args.timeout = timeout_s
    args.data_parallel_size = data_parallel_size
    args.eval_parallel_size = eval_parallel_size
    args.iteration_num = iteration_num
    args.batch_size = batch_size
    args.task_selection_mode = selection_mode
    args._loaded_config_payload = config_payload
    args.baseline_cpp = str(baseline_cpp)
    args.template_cpp = str(template_cpp)
    args.results_root = str(results_root)
    args.temp_root = str(temp_root)

    primary_label = str(config_payload.get("primary_label", "colors"))
    secondary_label = str(config_payload.get("secondary_label", "par2"))
    primary_objective = str(config_payload.get("primary_objective", "min")).lower()
    enable_exploration = bool(config_payload.get("enable_exploration", False))
    exploration_timeout = float(config_payload.get("exploration_timeout", 5.0) or 5.0)

    train_instances = _load_instances(train_data_dir)
    eval_instances = _load_instances(eval_data_dir)
    scale = PlotScale(
        time_max=max(timeout_s, 1.0),
        color_max=float(sum(instance.n for instance in train_instances)),
        iteration_max=max(iteration_num, 1),
    )

    checkpoint_dir = results_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if bool(config_payload.get("enable_server", False)):
        server_port = int(config_payload.get("server_port", getattr(args, "server_port", 8080)) or 8080)
        try:
            start_server(port=server_port, progress_file=str(results_root / "progress.json"))
        except OSError as exc:
            _log(f"[Server] Could not start on port {server_port}: {exc}. Continuing without dashboard.")

    if bool(config_payload.get("resume_from_checkpoint", True)):
        checkpoint = _load_checkpoint(checkpoint_dir)
    else:
        checkpoint = None

    # Initialize template with baseline code (markers preserved) if not already present.
    if not TEMPLATE_CPP.exists() or not any(f"<--{n}-->" in TEMPLATE_CPP.read_text(encoding="utf-8") for n in task_names):
        TEMPLATE_CPP.parent.mkdir(parents=True, exist_ok=True)
        TEMPLATE_CPP.write_text(adapter.render_source({}, keep_markers=True), encoding="utf-8")

    baseline_text = TEMPLATE_CPP.read_text(encoding="utf-8")
    baseline_codes = {name: adapter.extract_baseline_section(name, baseline_text=baseline_text) for name in task_names}

    baseline_summary, baseline_runs, baseline_state = _baseline_result(
        adapter=adapter,
        task_specs=task_specs,
        baseline_codes=baseline_codes,
        prompt_path=prompt_original,
        args=args,
        results_root=results_root,
        temp_root=temp_root,
        instances=train_instances,
        scale=scale,
        max_workers=data_parallel_size,
        source_text=baseline_text,
    )

    _baseline_trace_runs = [
        {
            "name": run.name,
            "final_colors": run.final_colors,
            "final_time": run.final_time,
            "final_timed_out": run.timed_out,
            "final_par2": run.par2,
            "trace": run.trace,
        }
        for run in baseline_runs
    ]
    best_state: dict[str, Any] = {
        "score": baseline_summary.get("score", [10**9, float("inf")]),
        "summary": baseline_summary,
        "updates": {},
        "rendered_source": baseline_state["source"],
        "trace_runs": _baseline_trace_runs,
    }
    global_best_state: dict[str, Any] = dict(best_state)

    iterations_log: list[dict[str, Any]] = []
    all_iteration_runs: list[list[dict[str, Any]]] = [best_state["trace_runs"]]

    def _summary_to_best_record(summary: Mapping[str, Any], iter_idx: int) -> dict[str, Any]:
        return {
            "iter": iter_idx,
            "total_colors": summary.get("total_colors", float("inf")),
            "par2": float(summary.get("par2", float("inf"))),
            "total_time": float(summary.get("total_time", float("inf"))),
        }

    best_by_primary: dict[str, Any] = _summary_to_best_record(baseline_summary, 0)
    best_by_secondary: dict[str, Any] = _summary_to_best_record(baseline_summary, 0)
    exploration_state: dict[str, Any] = _load_exploration_stats(results_root)

    if checkpoint and checkpoint.get("best_state"):
        best_state = checkpoint["best_state"]
        global_best_state = checkpoint.get("global_best_state", dict(best_state))
        best_by_primary = checkpoint.get("best_by_primary", best_by_primary)
        best_by_secondary = checkpoint.get("best_by_secondary", best_by_secondary)
        iterations_log = list(checkpoint.get("iterations_log", []))
        all_iteration_runs = list(checkpoint.get("all_iteration_runs", all_iteration_runs))

    _write_progress(
        results_root,
        _progress_payload(run_id, baseline_summary, best_state, iterations_log, primary_label, secondary_label),
    )

    if not checkpoint:
        _save_checkpoint(
            checkpoint_dir,
            {
                "run_id": run_id,
                "next_iteration": 0,
                "best_state": best_state,
                "global_best_state": global_best_state,
                "best_by_primary": best_by_primary,
                "best_by_secondary": best_by_secondary,
                "iterations_log": iterations_log,
                "all_iteration_runs": all_iteration_runs,
            },
        )

    for iteration_index in range(int(checkpoint.get("next_iteration", 0)) if checkpoint else 0, iteration_num):
        _log(f"Iteration {iteration_index + 1}/{iteration_num} start")
        current_mode = _task_for_iteration(task_names, selection_mode, iteration_index, rand_seed)
        all_tasks_mode = current_mode == "__all__"
        current_task = task_names[0] if all_tasks_mode else current_mode

        prompt_path = results_root / "prompts" / f"iter_{iteration_index:04d}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        last_iter_result: dict[str, Any] | None = None
        if iterations_log:
            last_iter_result = iterations_log[-1].get("best_prompt_context")

        exploration_stats_section = _format_exploration_stats_section(exploration_state.get("accumulated", {})) if enable_exploration else ""
        best_results_section = _format_best_results_section(best_by_primary, best_by_secondary, primary_label, secondary_label, primary_objective) if iterations_log else ""

        current_template_text = TEMPLATE_CPP.read_text(encoding="utf-8")
        current_codes = {
            name: adapter.extract_baseline_section(name, baseline_text=current_template_text) 
            for name in task_names
        }

        prompt_text = _build_prompt(
            prompt_feedback if iterations_log else prompt_original,
            task_specs,
            current_codes,
            baseline_info={
                "colors": baseline_summary.get("total_colors"),
                "time": baseline_summary.get("total_time"),
                "timed_out": baseline_summary.get("timed_out"),
                "par2": baseline_summary.get("par2"),
            },
            last_iter_result=last_iter_result,
            structured_output=structured_output,
            all_tasks_mode=all_tasks_mode,
            exploration_stats_section=exploration_stats_section,
            best_results_section=best_results_section,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")

        candidate_payloads: list[dict[str, Any]] = []
        for candidate_index in range(batch_size):
            if all_tasks_mode:
                payload = _query_llm_all(prompt_path, args, task_names, enable_exploration=enable_exploration)
            else:
                payload = _query_llm(prompt_path, args, current_task)
            candidate_payloads.append({"candidate_index": candidate_index, "payload": payload})

        best_candidate: dict[str, Any] | None = None
        best_runs: list[InstanceRun] = []
        best_rendered = best_state["rendered_source"]

        _template_base = TEMPLATE_CPP.read_text(encoding="utf-8")

        for candidate in candidate_payloads:
            candidate_index = int(candidate["candidate_index"])
            _log(f"Iteration {iteration_index + 1}: candidate {candidate_index + 1}/{batch_size}")
            payload = candidate["payload"]
            updates = _render_candidate_source(adapter, task_names, payload, all_tasks_mode)
            rendered_source = adapter.render_source(updates, baseline_text=_template_base)
            candidate_cpp = temp_root / f"iter_{iteration_index:04d}" / f"candidate_{candidate_index:02d}" / "fcns.cpp"
            candidate_cpp.parent.mkdir(parents=True, exist_ok=True)
            candidate_cpp.write_text(rendered_source, encoding="utf-8")
            candidate_exe = candidate_cpp.with_suffix("")

            summary, runs = _evaluate_source(
                candidate_cpp,
                candidate_exe,
                train_instances,
                timeout_s,
                max_workers=data_parallel_size,
            )
            _save_candidate_artifacts(results_root, iteration_index, candidate_index, summary, runs, rendered_source)

            candidate_state = {
                "candidate_index": candidate_index,
                "payload": payload,
                "updates": updates,
                "summary": summary,
                "rendered_source": rendered_source,
                "trace_runs": [
                    {
                        "name": run.name,
                        "final_colors": run.final_colors,
                        "final_time": run.final_time,
                        "trace": run.trace,
                        "final_timed_out": run.timed_out,
                        "final_par2": run.par2,
                        "greedy_valid": run.greedy_valid,
                    }
                    for run in runs
                ],
            }

            if best_candidate is None or _select_better(best_candidate["summary"], summary):
                best_candidate = candidate_state
                best_runs = runs
                best_rendered = rendered_source

        assert best_candidate is not None
        best_state = {
            "score": best_candidate["summary"]["score"],
            "summary": best_candidate["summary"],
            "updates": best_candidate["updates"],
            "rendered_source": best_candidate["rendered_source"],
            "trace_runs": best_candidate["trace_runs"],
        }
        global_improved = _select_better(global_best_state["summary"], best_candidate["summary"])
        if global_improved:
            global_best_state = dict(best_state)

        s = best_candidate["summary"]
        if float(s.get("total_colors", float("inf"))) < float(best_by_primary.get("total_colors", float("inf"))):
            best_by_primary = _summary_to_best_record(s, iteration_index + 1)
        if float(s.get("par2", float("inf"))) < float(best_by_secondary.get("par2", float("inf"))):
            best_by_secondary = _summary_to_best_record(s, iteration_index + 1)

        if enable_exploration:
            exp_code = str(best_candidate["payload"].get("__exploration__", {}).get("code", "") if all_tasks_mode else "")
            if exp_code.strip():
                new_stats = _execute_exploration_code(exp_code, train_instances, exploration_timeout)
                if new_stats:
                    accumulated = dict(exploration_state.get("accumulated", {}))
                    accumulated.update(new_stats)
                    iter_entry = {"iter": iteration_index, "stat_keys": sorted(new_stats.keys())}
                    exploration_state = {
                        "accumulated": accumulated,
                        "iterations": list(exploration_state.get("iterations", [])) + [iter_entry],
                    }
                    _atomic_write_json(results_root / "exploration_stats.json", exploration_state)

        _best_payload = best_candidate["payload"]
        if all_tasks_mode:
            _first_impl = _best_payload.get(task_names[0], {}) if task_names else {}
        else:
            _first_impl = _best_payload
        _first_impl = _first_impl if isinstance(_first_impl, Mapping) else {}
        iterations_log.append(
            {
                "iteration": iteration_index,
                "iter": iteration_index,
                "task_mode": current_mode,
                "task": current_mode,
                "title": str(_first_impl.get("title", "") or ""),
                "reason": str(_first_impl.get("reason", "") or ""),
                "best_summary": best_candidate["summary"],
                "best_updates": best_candidate["updates"],
                "best_code": best_candidate["rendered_source"],
                "comparison_code": best_candidate["rendered_source"],
                "code_preview": best_candidate["rendered_source"][:200],
                "best_prompt_context": _iter_prompt_context(
                    task_names,
                    best_candidate["payload"],
                    best_candidate["rendered_source"],
                    all_tasks_mode,
                    best_candidate["summary"],
                ),
            }
        )
        all_iteration_runs.append(best_candidate["trace_runs"])

        save_iteration_plots(
            results_root,
            iteration_index + 1,
            best_candidate["trace_runs"],
            scale,
            run_label=run_id,
        )

        TEMPLATE_CPP.parent.mkdir(parents=True, exist_ok=True)
        TEMPLATE_CPP.write_text(
            adapter.render_source(global_best_state["updates"], keep_markers=True),
            encoding="utf-8",
        )

        _write_progress(
            results_root,
            {
                **_progress_payload(run_id, baseline_summary, best_state, iterations_log, primary_label, secondary_label),
                "iteration": iteration_index,
            },
        )
        _save_checkpoint(
            checkpoint_dir,
            {
                "run_id": run_id,
                "next_iteration": iteration_index + 1,
                "best_state": best_state,
                "global_best_state": global_best_state,
                "best_by_primary": best_by_primary,
                "best_by_secondary": best_by_secondary,
                "iterations_log": iterations_log,
                "all_iteration_runs": all_iteration_runs,
            },
        )

    save_overall_3d_plot(results_root, all_iteration_runs, scale, run_label=run_id)

    final_path = results_root / "final_result.json"
    _atomic_write_json(
        final_path,
        {
            "run_id": run_id,
            "best_summary": best_state["summary"],
            "best_updates": best_state["updates"],
            "iterations": iterations_log,
        },
    )

    if bool(config_payload.get("run_eval", True)) and eval_instances:
        _log("Running eval evaluation")
        _evaluate_best_on_eval(
            adapter=adapter,
            rendered_source=str(best_state["rendered_source"]),
            eval_instances=eval_instances,
            results_root=results_root,
            temp_root=temp_root,
            timeout_s=timeout_s,
            max_workers=eval_parallel_size,
        )

    print(f"[FCNS] Run complete: {results_root}", flush=True)


def _apply_env_overrides(args: argparse.Namespace) -> argparse.Namespace:
    args.llm_model = os.getenv("AUTOSAT_LLM_MODEL", getattr(args, "llm_model", ""))
    args.api_base = os.getenv("AUTOSAT_API_BASE", getattr(args, "api_base", ""))
    args.api_key = os.getenv("AUTOSAT_API_KEY", getattr(args, "api_key", ""))
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--iteration_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--task_selection_mode", type=str, default="all")
    parser.add_argument("--rand_seed", type=int, default=42)
    parser.add_argument("--llm_model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--eval_parallel_size", type=int, default=1)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=True)
    parser.add_argument("--run_eval", type=bool, default=True)
    parser.add_argument("--use_structured", type=bool, default=True)
    args = parser.parse_args()

    loaded_config = _load_config(Path(args.config))
    for key, value in loaded_config.items():
        setattr(args, key, value)
    args._loaded_config_payload = loaded_config
    args = _apply_env_overrides(args)

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    main(args)