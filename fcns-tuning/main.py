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
    trace: list[dict[str, Any]]
    stdout: str
    stderr: list[str]
    exit_code: int
    timed_out: bool


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


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
    if stdout_lines:
        try:
            final_colors = int(stdout_lines[0].split()[0])
        except Exception:
            pass
    if trace:
        color_points = [float(item["colors"]) for item in trace if isinstance(item.get("colors"), (int, float))]
        time_points = [float(item["elapsed"]) for item in trace if isinstance(item.get("elapsed"), (int, float))]
        if color_points:
            final_colors = int(color_points[-1])
        if time_points:
            final_time = max(time_points)

    return InstanceRun(
        name=instance.name,
        final_colors=final_colors,
        final_time=final_time,
        trace=trace,
        stdout=stdout_text,
        stderr=stderr_lines,
        exit_code=proc.returncode or 0,
        timed_out=timed_out,
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
    if worker_count > 1 and len(instances) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_count, len(instances))) as executor:
            runs = list(executor.map(lambda instance: _run_solver(executable_path, instance, timeout_s), instances))
    else:
        runs = [_run_solver(executable_path, instance, timeout_s) for instance in instances]

    total_colors = 0
    total_time = 0.0
    all_ok = True
    for run in runs:
        total_colors += int(run.final_colors)
        total_time += float(run.final_time)
        all_ok = all_ok and (not run.timed_out) and run.exit_code == 0

    return (
        {
            "compile_ok": True,
            "score": [total_colors, total_time],
            "all_ok": all_ok,
            "total_colors": total_colors,
            "total_time": total_time,
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
    )


def _query_llm(prompt_file: Path, args: argparse.Namespace, task_name: str) -> dict[str, str]:
    llm_api = get_llm_api(args)
    temperature = float(getattr(args, "temperature", 1.0) or 1.0)
    if getattr(llm_api, "_structured_output", False):
        return llm_api.call_api_structured(prompt_file=str(prompt_file), temperature=temperature, task_name=task_name)
    raw = llm_api.call_api(prompt_file=str(prompt_file), temperature=temperature)
    parsed = parse_structured_response(raw)
    return parsed


def _query_llm_all(prompt_file: Path, args: argparse.Namespace, task_names: Sequence[str]) -> dict[str, dict[str, str]]:
    llm_api = get_llm_api(args)
    temperature = float(getattr(args, "temperature", 1.0) or 1.0)
    prompt_text = prompt_file.read_text(encoding="utf-8")

    if getattr(llm_api, "_structured_output", False):
        if llm_api.__class__.__name__ == "GPTCallAPI":
            raw = llm_api.call_structured_all(prompt_text, temperature, build_structured_schema_all(task_names))
            return parse_multi_response(raw, task_names)
        if llm_api.__class__.__name__ == "ElizaCallAPI":
            raw = llm_api.call_structured_all(prompt_text, temperature, build_tool_schema_all(task_names))
            return parse_multi_response(raw, task_names)

    raw = llm_api.call_api(prompt_file=str(prompt_file), temperature=temperature)
    return parse_multi_response(raw, task_names)


def _select_better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_key = (float(left.get("total_colors", float("inf"))), float(left.get("total_time", float("inf"))))
    right_key = (float(right.get("total_colors", float("inf"))), float(right.get("total_time", float("inf"))))
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
) -> tuple[dict[str, Any], list[InstanceRun], dict[str, str]]:
    baseline_rendered = adapter.render_source({})
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
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "metric_mode": "colors_time",
        "metric_labels": {"primary": "colors", "secondary": "time"},
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
        return {"implementations": implementations}

    return {
        "code": str(payload.get("code", "") or ""),
        "title": str(payload.get("title", "") or ""),
        "reason": str(payload.get("reason", "") or ""),
    }


def _write_progress(results_root: Path, payload: Mapping[str, Any]) -> None:
    write_progress(payload, progress_file=str(results_root / "progress.json"))


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
    batch_size = int(config_payload.get("batch_size", getattr(args, "batch_size", 4)) or 4)
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
        start_server(port=server_port, progress_file=str(results_root / "progress.json"))

    if bool(config_payload.get("resume_from_checkpoint", True)):
        checkpoint = _load_checkpoint(checkpoint_dir)
    else:
        checkpoint = None

    baseline_text = adapter.baseline_text()
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
    )

    best_state: dict[str, Any] = {
        "score": baseline_summary.get("score", [10**9, float("inf")]),
        "summary": baseline_summary,
        "updates": {},
        "rendered_source": baseline_state["source"],
        "trace_runs": [
            {
                "name": run.name,
                "final_colors": run.final_colors,
                "final_time": run.final_time,
                "trace": run.trace,
            }
            for run in baseline_runs
        ],
    }

    iterations_log: list[dict[str, Any]] = []
    all_iteration_runs: list[list[dict[str, Any]]] = [best_state["trace_runs"]]

    if checkpoint and checkpoint.get("best_state"):
        best_state = checkpoint["best_state"]
        iterations_log = list(checkpoint.get("iterations_log", []))
        all_iteration_runs = list(checkpoint.get("all_iteration_runs", all_iteration_runs))

    _write_progress(
        results_root,
        _progress_payload(run_id, baseline_summary, best_state, iterations_log),
    )

    if not checkpoint:
        _save_checkpoint(
            checkpoint_dir,
            {
                "run_id": run_id,
                "next_iteration": 0,
                "best_state": best_state,
                "iterations_log": iterations_log,
                "all_iteration_runs": all_iteration_runs,
            },
        )

    for iteration_index in range(int(checkpoint.get("next_iteration", 0)) if checkpoint else 0, iteration_num):
        current_mode = _task_for_iteration(task_names, selection_mode, iteration_index, rand_seed)
        all_tasks_mode = current_mode == "__all__"
        current_task = task_names[0] if all_tasks_mode else current_mode

        prompt_path = results_root / "prompts" / f"iter_{iteration_index:04d}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        last_iter_result: dict[str, Any] | None = None
        if iterations_log:
            last_iter_result = iterations_log[-1].get("best_prompt_context")

        prompt_text = _build_prompt(
            prompt_feedback if iterations_log else prompt_original,
            task_specs,
            baseline_codes,
            baseline_info={
                "colors": best_state["summary"].get("total_colors"),
                "time": best_state["summary"].get("total_time"),
            },
            last_iter_result=last_iter_result,
            structured_output=structured_output,
            all_tasks_mode=all_tasks_mode,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")

        candidate_payloads: list[dict[str, Any]] = []
        for candidate_index in range(batch_size):
            if all_tasks_mode:
                payload = _query_llm_all(prompt_path, args, task_names)
            else:
                payload = _query_llm(prompt_path, args, current_task)
            candidate_payloads.append({"candidate_index": candidate_index, "payload": payload})

        best_candidate: dict[str, Any] | None = None
        best_runs: list[InstanceRun] = []
        best_rendered = best_state["rendered_source"]

        for candidate in candidate_payloads:
            candidate_index = int(candidate["candidate_index"])
            payload = candidate["payload"]
            updates = _render_candidate_source(adapter, task_names, payload, all_tasks_mode)
            rendered_source = adapter.render_source(updates)
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

        iterations_log.append(
            {
                "iteration": iteration_index,
                "task_mode": current_mode,
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

        if best_rendered != best_state["rendered_source"]:
            best_rendered = best_state["rendered_source"]
        TEMPLATE_CPP.parent.mkdir(parents=True, exist_ok=True)
        TEMPLATE_CPP.write_text(best_rendered, encoding="utf-8")

        _write_progress(
            results_root,
            {
                **_progress_payload(run_id, baseline_summary, best_state, iterations_log),
                "iteration": iteration_index,
            },
        )
        _save_checkpoint(
            checkpoint_dir,
            {
                "run_id": run_id,
                "next_iteration": iteration_index + 1,
                "best_state": best_state,
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