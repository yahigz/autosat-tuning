import os
import argparse
import ast
import json
import time
import uuid
import random
import yaml
import re
import sys
import glob
import signal
import subprocess
from pathlib import Path

from utils import get_code, revise_file, clean_files, collect_results, \
                  copy_folder, delete_InfiniteLoopInst, get_batch_id, train_init, check_reIteration
from llm_api.base_api import get_llm_api
from prompting import (
    build_prompt_text,
    load_prompt_text,
    parse_structured_response,
    parse_multi_response,
    get_code_from_text_all,
    build_tool_schema_all,
    build_structured_schema_all,
)
from execution.execution_worker import ExecutionWorker
from evaluation.evaluate import evaluate
from plotting import plot_training_curve
import warnings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOLVER_BASELINE_DIR = Path("solver/baseline")
SOLVER_BASELINE_CPP = SOLVER_BASELINE_DIR / "EasySAT.cpp"
SOLVER_TEMPLATE_DIR = Path("solver/template")
PROMPTS_DIR = Path("prompts")

_SHUTDOWN_IN_PROGRESS = False


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def _graceful_shutdown(reason="", exit_code=None):
    global _SHUTDOWN_IN_PROGRESS
    if _SHUTDOWN_IN_PROGRESS:
        if exit_code is not None:
            os._exit(int(exit_code))
        return
    _SHUTDOWN_IN_PROGRESS = True
    try:
        if reason:
            print(f"[Shutdown] {reason}", flush=True)
        ExecutionWorker.shutdown_all(timeout=2)
    except Exception:
        pass
    try:
        if os.name == 'posix':
            for pattern in ['EasySAT', 'SAT_Solver_tmp']:
                subprocess.run(['pkill', '-TERM', '-f', pattern], check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            for pattern in ['EasySAT', 'SAT_Solver_tmp']:
                subprocess.run(['pkill', '-KILL', '-f', pattern], check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
    except Exception:
        pass
    finally:
        if exit_code is not None:
            os._exit(int(exit_code))


# ---------------------------------------------------------------------------
# Baseline marker utilities
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(r'<--([A-Za-z_]\w*)-->')


def discover_available_tasks(baseline_cpp: Path = SOLVER_BASELINE_CPP) -> list[str]:
    """Return task names found as marker pairs in the baseline solver."""
    if not baseline_cpp.exists():
        raise FileNotFoundError(f"Baseline solver not found: {baseline_cpp}")
    text = baseline_cpp.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    tasks: list[str] = []
    for m in _MARKER_RE.finditer(text):
        name = m.group(1)
        seen[name] = seen.get(name, 0) + 1
    for name, count in seen.items():
        if count >= 2:
            tasks.append(name)
    return tasks


def extract_baseline_section(baseline_text: str, marker_name: str) -> str:
    """Return the code between the two <--marker_name--> markers."""
    pattern = re.compile(
        r'<--' + re.escape(marker_name) + r'-->\n(.*?)\n<--' + re.escape(marker_name) + r'-->',
        re.DOTALL,
    )
    m = pattern.search(baseline_text)
    return m.group(1).strip() if m else ""


def build_solver_source(
    codes: dict[str, str],
    timeout: int | str,
    data_dir: str,
    baseline_cpp: Path = SOLVER_BASELINE_CPP,
) -> str:
    """Build a complete solver C++ string.

    For each <--marker--> pair in the baseline:
      - if marker name is in `codes` and code is non-empty → use that code
      - otherwise → keep the baseline code between the markers
    All markers are removed from the output. {{ timeout }} and {{ data_dir }}
    placeholders are substituted.
    """
    text = baseline_cpp.read_text(encoding="utf-8")

    def _replace_section(m_name: str, replacement: str) -> None:
        nonlocal text
        pattern = re.compile(
            r'<--' + re.escape(m_name) + r'-->\n(.*?)\n<--' + re.escape(m_name) + r'-->',
            re.DOTALL,
        )
        found = pattern.search(text)
        if not found:
            return
        text = pattern.sub(replacement.strip(), text, count=1)

    for name in discover_available_tasks(baseline_cpp):
        provided = codes.get(name, "")
        if provided and len(provided.strip()) >= 10:
            _replace_section(name, provided.strip())
        else:
            baseline_code = extract_baseline_section(text, name)
            _replace_section(name, baseline_code)

    text = re.sub(r'\{\{\s*timeout\s*\}\}', str(timeout), text)
    text = re.sub(r'\{\{\s*data_dir\s*\}\}', str(data_dir), text)
    return text


def update_solver_template(codes: dict[str, str]) -> None:
    """Write solver/template/EasySAT.cpp with best-known code for every task."""
    SOLVER_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    for hdr in ("EasySAT.hpp", "heap.hpp"):
        src = SOLVER_BASELINE_DIR / hdr
        dst = SOLVER_TEMPLATE_DIR / hdr
        if src.exists() and not dst.exists():
            import shutil as _shutil
            _shutil.copy(src, dst)

    rendered = build_solver_source(codes, timeout="{{ timeout }}", data_dir="{{ data_dir }}")
    # Re-insert generic placeholders so the template file stays renderable
    (SOLVER_TEMPLATE_DIR / "EasySAT.cpp").write_text(rendered, encoding="utf-8")
    print(f"[Template] Updated solver/template/EasySAT.cpp ({len(codes)} heuristics)", flush=True)


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------
ALL_TASKS_SENTINEL = "__all__"


def _task_for_iteration(task_sequence: list[str], selection_mode: str, iter_idx: int, rand_seed: int = 42) -> str:
    """Return the task name for this iteration, or ALL_TASKS_SENTINEL when mode is 'all'."""
    if not task_sequence:
        raise ValueError("task_sequence must not be empty")
    mode = str(selection_mode or "random_one").strip().lower()
    if mode == "all":
        return ALL_TASKS_SENTINEL
    if mode == "random_one":
        return random.Random(int(rand_seed)).choice(task_sequence)
    if mode in {"cycle"}:
        return task_sequence[iter_idx % len(task_sequence)]
    raise ValueError(f"task_selection_mode must be one of: all, random_one, cycle. Got: {mode}")


# ---------------------------------------------------------------------------
# Paths / run layout
# ---------------------------------------------------------------------------
def _make_run_id(explicit: str = "") -> str:
    explicit = str(explicit or "").strip()
    return explicit or (time.strftime("run_%Y%m%d_%H%M%S") + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}")


def _run_paths(run_id: str) -> dict:
    temp_root = Path("./temp/runs") / run_id
    results_root = Path("./results/runs") / run_id
    paths = {
        "run_id": run_id,
        "temp_root": temp_root,
        "temp_results_dir": Path("./temp/results"),
        "temp_prompts_dir": temp_root / "prompts",
        "temp_easy_root": temp_root / "EasySAT",
        "results_root": results_root,
        "checkpoint_dir": results_root / "checkpoints",
        "snapshots_dir": results_root / "snapshots",
        "eval_results_dir": results_root / "eval_results",
    }
    for key in ("temp_prompts_dir", "temp_easy_root", "checkpoint_dir", "snapshots_dir", "eval_results_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["temp_results_dir"].mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_checkpoint(next_iter: int, results: dict, answers: dict, extra_params: dict,
                     best_result: dict, checkpoint_dir: Path, run_id: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "run_id": run_id,
        "next_iter": next_iter,
        "results": results,
        "answers": {str(k): v for k, v in answers.items()},
        "extra_params": {str(k): v for k, v in extra_params.items()},
        "best_result": best_result,
        "saved_at": time.time(),
    }
    _atomic_write_json(checkpoint_dir / "latest_checkpoint.json", state)
    _atomic_write_json(checkpoint_dir / f"iter_{next_iter - 1}_checkpoint.json", state)


def _load_checkpoint(checkpoint_dir: Path) -> dict | None:
    latest = checkpoint_dir / "latest_checkpoint.json"
    if not latest.exists():
        return None
    try:
        state = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.warn(f"Checkpoint corrupted: {latest} — {exc}", stacklevel=2)
        return None
    state["answers"] = {int(k): v for k, v in state.get("answers", {}).items()}
    state["extra_params"] = {int(k): v for k, v in state.get("extra_params", {}).items()}
    return state


def _save_iteration_artifacts(iter_idx: int, result: dict, best_result: dict,
                               temp_prompts_dir: Path, results_root: Path, snapshots_dir: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    for d in (temp_prompts_dir, results_root, snapshots_dir):
        d.mkdir(parents=True, exist_ok=True)
    for d in (temp_prompts_dir, results_root):
        (d / f"iter_{iter_idx}_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if best_result:
        best_id = next(iter(best_result))
        bd = best_result[best_id]
        (snapshots_dir / f"iter_{iter_idx}_best.json").write_text(
            json.dumps({"iter": iter_idx, "best_id": best_id,
                        "time": bd[0], "code": bd[1], "PAR-2": bd[2]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------
def _strip_start_end_markers(code_text: str) -> str:
    text = str(code_text or "").strip()
    m = re.search(r"// start\n([\s\S]*?)\n// end", text, re.MULTILINE)
    return m.group(1).strip() if m else text


def _looks_like_code(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    banned = ["must start with", "Tips:", "''' and end with '''"]
    low = text.lower()
    if any(b.lower() in low for b in banned):
        return False
    return any(tok in text for tok in ["void Solver::", "else if", "restart();", "{", ";"])


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def _load_prompt_file(mode: str) -> str:
    filename = "original_prompt.txt" if mode == "original" else "feedback_prompt.txt"
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_prompt(
    mode: str,
    task_name: str,
    baseline_par2: float | None,
    baseline_code: str | dict,
    last_iter_result: dict | None,
    use_structured: bool,
    config_payload: dict,
    output_path: Path,
    all_tasks_mode: bool = False,
    task_sequence: list[str] | None = None,
) -> str:
    base_text = _load_prompt_file(mode)
    prompt = build_prompt_text(
        base_prompt_text=base_text,
        task_name=task_name,
        baseline_par2=baseline_par2,
        baseline_code=baseline_code,
        last_iter_result=last_iter_result,
        config_payload=config_payload,
        structured_output=use_structured,
        allowed_modules=task_sequence if all_tasks_mode else [task_name],
        all_tasks_mode=all_tasks_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    return str(output_path)


# ---------------------------------------------------------------------------
# Solver source rendering
# ---------------------------------------------------------------------------
def _render_solver_source(
    target_file: str,
    task_name: str,
    answer_code: str,
    timeout: int,
    data_dir: str,
    extra_codes: dict[str, str] | None = None,
) -> None:
    chosen = _strip_start_end_markers(answer_code)
    if len(chosen.strip()) < 10:
        chosen = extract_baseline_section(SOLVER_BASELINE_CPP.read_text(encoding="utf-8"), task_name)
    if len(chosen.strip()) < 10:
        raise ValueError(f"No valid code to inject for task={task_name}")

    codes = dict(extra_codes or {})
    codes[task_name] = chosen

    rendered = build_solver_source(codes, timeout=timeout, data_dir=f'"{data_dir}"')
    Path(target_file).parent.mkdir(parents=True, exist_ok=True)
    Path(target_file).write_text(rendered, encoding="utf-8")


# ---------------------------------------------------------------------------
# Validate (compile check)
# ---------------------------------------------------------------------------
def _validate_solver(source_path: str) -> tuple[bool, str]:
    out = Path(source_path).with_suffix(".compile_check")
    proc = subprocess.run(
        ["g++", "-O3", "-Wall", "-std=c++17", source_path, "-o", str(out)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        out.unlink(missing_ok=True)
    except Exception:
        pass
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _query_llm(prompt_file: str, count: int, args) -> tuple[int, dict]:
    llm_api = get_llm_api(args)
    temperature = getattr(args, "temperature", 1.0)
    use_structured = getattr(llm_api, "_structured_output", False)

    if use_structured:
        structured = llm_api.call_api_structured(
            prompt_file=prompt_file, temperature=temperature, task_name=getattr(args, "task", "")
        )
        code = str(structured.get("code", "") or "")
        title = str(structured.get("title", "") or "")
        reason = str(structured.get("reason", "") or "")
    else:
        raw = llm_api.call_api(prompt_file=prompt_file, temperature=temperature)
        code = get_code(raw, seperator=["// start\n", "\n// end"])
        title = ""
        reason = ""

    if code.strip().startswith("{"):
        parsed = parse_structured_response(code)
        if parsed.get("code"):
            code = parsed["code"]
            title = parsed.get("title", title)
            reason = parsed.get("reason", reason)

    return count, {"code": code.strip(), "title": title.strip(), "reason": reason.strip()}


def _execute_candidate(count: int, results: dict, args, answer_payload: dict, extra_codes: dict) -> tuple[int, bool, str]:
    worker = ExecutionWorker()
    code = answer_payload.get("code", "") if isinstance(answer_payload, dict) else str(answer_payload or "")

    if args.devoid_duplication and code in results["prompt"].values():
        return count, False, code

    save_path = os.path.join(args.temp_root, f"EasySAT_{(count - 1) % args.batch_size}", "EasySAT.cpp")
    _render_solver_source(
        target_file=save_path,
        task_name=args.task,
        answer_code=code,
        timeout=args.timeout,
        data_dir=args.data_dir,
        extra_codes=extra_codes,
    )
    success = worker.execute(count, args.batch_size, args.data_parallel_size)
    return count, bool(success), code


# ---------------------------------------------------------------------------
# Multi-task (all-mode) LLM call and execution
# ---------------------------------------------------------------------------

def _query_llm_all(prompt_file: str, count: int, args, task_sequence: list[str]) -> tuple[int, dict[str, dict]]:
    """Query LLM for all tasks at once. Returns {task_name: {code, title, reason}}."""
    llm_api = get_llm_api(args)
    temperature = getattr(args, "temperature", 1.0)
    use_structured = getattr(llm_api, "_structured_output", False)

    if use_structured:
        # Use multi-task structured schema
        if hasattr(llm_api, "_call_structured_raw"):
            with open(prompt_file, encoding="utf-8") as f:
                prompt = f.read()
            # Eliza: use tool_use for all tasks
            if hasattr(llm_api, "_post"):
                tool_schema = build_tool_schema_all(task_sequence)
                payload = {
                    "model": llm_api.model_name,
                    "max_tokens": llm_api._DEFAULT_MAX_TOKENS,
                    "system": llm_api._build_structured_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "tools": [tool_schema],
                    "tool_choice": {"type": "any"},
                }
                try:
                    data = llm_api._call_with_retries(lambda: llm_api._post(payload), description="all-tasks[eliza]")
                    llm_api._log_usage_from_response(data)
                    for block in data.get("content", []):
                        if block.get("type") == "tool_use":
                            raw = json.dumps(block.get("input", {}))
                            return count, parse_multi_response(raw, task_sequence)
                except Exception as exc:
                    print(f"[all-tasks] Eliza tool_use failed: {exc}, falling back", flush=True)
            # GPT-compatible: use json_schema for all tasks
            else:
                import openai as _openai
                schema = build_structured_schema_all(task_sequence)
                try:
                    resp = llm_api._call_with_retries(
                        lambda: _openai.ChatCompletion.create(
                            model=llm_api.model_name,
                            messages=[
                                {"role": "system", "content": llm_api._build_structured_system_prompt()},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=temperature,
                            stream=False,
                            response_format=schema,
                        ),
                        description="all-tasks[openai]",
                    )
                    llm_api._log_token_usage(resp.get("usage"))
                    raw = resp["choices"][0]["message"]["content"]
                    return count, parse_multi_response(raw, task_sequence)
                except Exception as exc:
                    print(f"[all-tasks] OpenAI structured failed: {exc}, falling back", flush=True)

    # Plain text fallback
    raw_text = llm_api.call_api(prompt_file=prompt_file, temperature=temperature)
    return count, get_code_from_text_all(raw_text, task_sequence)


def _execute_candidate_all(
    count: int,
    results: dict,
    args,
    payloads: dict[str, dict],
    task_sequence: list[str],
) -> tuple[int, bool, str]:
    """Build solver with codes for all tasks and execute it."""
    worker = ExecutionWorker()
    codes = {t: p["code"] for t, p in payloads.items() if p.get("code") and len(p["code"].strip()) >= 10}
    combined_repr = json.dumps(payloads, ensure_ascii=False)

    if args.devoid_duplication and combined_repr in results["prompt"].values():
        return count, False, combined_repr

    save_path = os.path.join(args.temp_root, f"EasySAT_{(count - 1) % args.batch_size}", "EasySAT.cpp")
    rendered = build_solver_source(codes, timeout=args.timeout, data_dir=f'"{args.data_dir}"')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(rendered, encoding="utf-8")

    success = worker.execute(count, args.batch_size, args.data_parallel_size)
    return count, bool(success), combined_repr


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _latest_result_file(results_dir: str, method_name: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(results_dir, f"results_{method_name}_*.txt")))
    return matches[-1] if matches else None


def _read_eval_par2(result_file: str) -> float:
    last = ""
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line.strip()
    payload = ast.literal_eval(last)
    return float(payload["PAR-2"])


def _evaluate_candidate(args, task_name: str, method_name: str, answer_code: str) -> tuple[float, str]:
    SAT_folder = os.path.join(args.temp_root, f"EasySAT_{method_name}")
    copy_folder(src_folder=args.temp_easy_root, num=1, mode="eval", target_folder=SAT_folder)
    cpp_path = os.path.join(SAT_folder, "EasySAT_modified.cpp")
    _render_solver_source(
        target_file=cpp_path,
        task_name=task_name,
        answer_code=answer_code,
        timeout=args.eval_timeout,
        data_dir=args.eval_data_dir,
    )
    ok, err = _validate_solver(cpp_path)
    if not ok:
        raise RuntimeError(err)
    prev_task = args.task
    try:
        args.task = task_name
        evaluate(args, method_name=method_name, SAT_solver_file_path=cpp_path)
    finally:
        args.task = prev_task
    result_file = _latest_result_file(args.results_save_path, method_name)
    if not result_file:
        raise FileNotFoundError(f"No eval result for {method_name}")
    return _read_eval_par2(result_file), cpp_path


def _run_eval_stage(args, final: dict, extra_params: dict, paths: dict) -> None:
    print("[Eval] Starting evaluation stage...", flush=True)
    if os.path.exists(args.temp_results_dir):
        clean_files(folder_path=args.temp_results_dir, mode="all")

    baseline_par2 = final["0"]["PAR-2"]
    print(f"[Eval] Baseline PAR-2: {baseline_par2}", flush=True)

    # Collect all candidates that beat the baseline (by train PAR-2)
    candidates = []
    for gid_str, item in final.items():
        if gid_str == "0":
            continue
        par2 = item.get("PAR-2")
        code = item.get("prompt", "")
        if par2 is not None and par2 < baseline_par2:
            candidates.append((int(gid_str), par2, code, extra_params.get(str(gid_str), {})))

    if not candidates:
        print("[Eval] No candidates beat baseline on train set.", flush=True)
        return

    # Best by train PAR-2
    best_gid, best_train_par2, best_code, best_meta = min(candidates, key=lambda x: x[1])
    print(f"[Eval] Best train candidate: global_id={best_gid} train_PAR-2={best_train_par2:.2f} "
          f"title={best_meta.get('title', '?')!r}", flush=True)

    model_tag = str(getattr(args, "llm_model", "model") or "model").replace("/", "")
    method_name = f"best_{best_gid}_{model_tag}"

    # Detect all-tasks-mode candidate: best_code is a JSON dict string
    is_all_mode_candidate = False
    all_codes: dict[str, str] = {}
    try:
        parsed = json.loads(best_code)
        if isinstance(parsed, dict) and any(isinstance(v, dict) for v in parsed.values()):
            is_all_mode_candidate = True
            all_codes = {t: info.get("code", "") for t, info in parsed.items() if isinstance(info, dict)}
    except (json.JSONDecodeError, TypeError):
        pass

    if is_all_mode_candidate:
        # Build a combined solver with all codes and evaluate it
        cpp_path = os.path.join(args.temp_root, f"EasySAT_eval_all_{best_gid}", "EasySAT_modified.cpp")
        rendered = build_solver_source(all_codes, timeout=args.eval_timeout, data_dir=f'"{args.eval_data_dir}"')
        Path(cpp_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cpp_path).write_text(rendered, encoding="utf-8")
        ok, err = _validate_solver(cpp_path)
        if not ok:
            warnings.warn(f"[Eval] All-mode solver compile failed: {err}", stacklevel=2)
            return
        prev_task = args.task
        try:
            evaluate(args, method_name=method_name, SAT_solver_file_path=cpp_path)
        finally:
            args.task = prev_task
        result_file = _latest_result_file(args.results_save_path, method_name)
        if not result_file:
            warnings.warn("[Eval] No result file for all-mode candidate.", stacklevel=2)
            return
        eval_par2 = _read_eval_par2(result_file)
        impls = best_meta.get("implementations", [])
        print(f"\n[Eval] === Best configuration (all-tasks mode) ===", flush=True)
        print(f"[Eval] global_id     : {best_gid}", flush=True)
        print(f"[Eval] train PAR-2   : {best_train_par2:.2f} (baseline: {baseline_par2:.2f})", flush=True)
        print(f"[Eval] eval  PAR-2   : {eval_par2:.2f}", flush=True)
        print(f"[Eval] eval delta    : {eval_par2 - baseline_par2:+.2f} vs baseline", flush=True)
        for impl in impls:
            print(f"[Eval]   {impl.get('task','?')}: {impl.get('title','')}", flush=True)
        _atomic_write_json(
            Path(args.results_root) / "eval_best_result.json",
            {
                "global_id": best_gid, "mode": "all",
                "implementations": impls,
                "train_PAR-2": best_train_par2, "eval_PAR-2": eval_par2,
                "baseline_PAR-2": baseline_par2,
            },
        )
        print(f"[Eval] Saved to {args.results_root}/eval_best_result.json", flush=True)
        return

    # Single-task candidate
    task_name = _infer_task_from_code(best_code) or getattr(args, "task", "")
    if not task_name:
        print("[Eval] Cannot infer task for best candidate — skipping.", flush=True)
        return

    try:
        eval_par2, _ = _evaluate_candidate(args, task_name, method_name, best_code)
        print(f"\n[Eval] === Best configuration ===", flush=True)
        print(f"[Eval] global_id     : {best_gid}", flush=True)
        print(f"[Eval] task          : {task_name}", flush=True)
        print(f"[Eval] title         : {best_meta.get('title', '')}", flush=True)
        print(f"[Eval] reason        : {best_meta.get('reason', '')}", flush=True)
        print(f"[Eval] train PAR-2   : {best_train_par2:.2f} (baseline: {baseline_par2:.2f})", flush=True)
        print(f"[Eval] eval  PAR-2   : {eval_par2:.2f}", flush=True)
        print(f"[Eval] eval delta    : {eval_par2 - baseline_par2:+.2f} vs baseline", flush=True)
        _atomic_write_json(
            Path(args.results_root) / "eval_best_result.json",
            {
                "global_id": best_gid, "task": task_name,
                "title": best_meta.get("title", ""), "reason": best_meta.get("reason", ""),
                "train_PAR-2": best_train_par2, "eval_PAR-2": eval_par2,
                "baseline_PAR-2": baseline_par2,
                "code": best_code,
            },
        )
        print(f"[Eval] Saved to {args.results_root}/eval_best_result.json", flush=True)
    except Exception as exc:
        warnings.warn(f"[Eval] Evaluation failed: {exc}", stacklevel=2)


def _infer_task_from_code(code: str) -> str | None:
    text = str(code or "").strip()
    m = re.search(r"void\s+Solver::([A-Za-z_]\w*)\s*\(", text)
    if m:
        fn = m.group(1)
        return {"restart": "restart_function", "rephase": "rephase_function", "bump_var": "bump_var_function"}.get(fn)
    if "restart();" in text:
        return "restart_condition"
    return None


# ---------------------------------------------------------------------------
# Env / config helpers
# ---------------------------------------------------------------------------
def _load_env_file():
    for path in [".env", "../.env", "../../.env"]:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        break


def _apply_env_overrides(args):
    _load_env_file()
    args.llm_model = os.getenv("AUTOSAT_LLM_MODEL", os.getenv("DEEPINFRA_MODEL", args.llm_model))
    args.api_base  = os.getenv("AUTOSAT_API_BASE",  os.getenv("DEEPINFRA_API_BASE", args.api_base))
    args.api_key   = os.getenv("AUTOSAT_API_KEY",   os.getenv("DEEPINFRA_API_KEY", args.api_key))
    return args


def _normalize_original_result(args):
    r = getattr(args, "original_result", {})
    if isinstance(r, dict):
        t, p = r.get("time", 0), r.get("PAR-2", 0)
    else:
        t, p = 0, r or 0
    args.original_result = {"time": t or 0, "PAR-2": p or 0}
    return args.original_result


# ---------------------------------------------------------------------------
# Progress server helpers
# ---------------------------------------------------------------------------
def _update_progress(args, run_id: str, baseline_par2: float | None, iterations_log: list[dict]) -> None:
    try:
        from server import write_progress
        write_progress(
            run_id=run_id,
            baseline_par2=baseline_par2,
            iterations=iterations_log,
            progress_file=str(Path(args.results_root) / "progress.json"),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Template write-back (greedy / annealing)
# ---------------------------------------------------------------------------
def _should_writeback(strategy: str, best_par2: float, prev_par2: float,
                      baseline_par2: float, iter_idx: int, n_iters: int, rng: random.Random) -> tuple[bool, str]:
    if strategy == "greedy_train":
        accept = best_par2 < prev_par2
        return accept, f"{'improvement' if accept else 'no improvement'} ({best_par2:.0f} vs prev {prev_par2:.0f})"
    if strategy == "annealing":
        import math
        T = 1.0 * (0.1 / 1.0) ** (iter_idx / max(n_iters - 1, 1))
        delta = best_par2 - prev_par2
        obj = delta / (0.25 * baseline_par2) if baseline_par2 > 0 else delta
        if obj < 0:
            return True, f"SA improvement (obj={obj:.4f})"
        prob = math.exp(-obj / T) if T > 0 else 0.0
        accept = rng.random() < prob
        return accept, f"SA Metropolis prob={prob:.4f} (obj={obj:.4f}, T={T:.4f}) → {'accepted' if accept else 'rejected'}"
    return False, "strategy=none"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main(args):
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
    _normalize_original_result(args)

    # Discover tasks from baseline markers
    available_tasks = discover_available_tasks()

    raw_tasks = getattr(args, "optimize_tasks", None)
    if isinstance(raw_tasks, str):
        raw_tasks = [t.strip() for t in raw_tasks.split(",") if t.strip()]
    task_sequence = [t for t in (raw_tasks or []) if t in available_tasks]
    if not task_sequence:
        fallback = getattr(args, "task", "")
        if fallback and fallback in available_tasks:
            task_sequence = [fallback]
        else:
            task_sequence = available_tasks[:1]
    if not task_sequence:
        raise ValueError("No valid tasks found. Check solver/baseline/EasySAT.cpp for markers.")

    selection_mode = str(getattr(args, "task_selection_mode", "random_one") or "random_one").lower()
    rand_seed = int(getattr(args, "rand_seed", 42) or 42)
    run_id = _make_run_id(getattr(args, "run_id", ""))
    os.environ["AUTOSAT_RUN_ID"] = run_id
    paths = _run_paths(run_id)

    args.run_id = run_id
    args.temp_root = str(paths["temp_root"])
    args.temp_results_dir = str(paths["temp_results_dir"])
    args.temp_prompts_dir = str(paths["temp_prompts_dir"])
    args.temp_easy_root = str(paths["temp_easy_root"])
    args.results_root = str(paths["results_root"])
    args.checkpoint_dir = str(paths["checkpoint_dir"])
    args.results_save_path = str(paths["eval_results_dir"])

    # Optional HTTP server
    if bool(getattr(args, "enable_server", False)):
        port = int(getattr(args, "server_port", 8080) or 8080)
        try:
            from server import start_server
            start_server(port=port, progress_file=str(Path(args.results_root) / "progress.json"))
        except Exception as e:
            warnings.warn(f"[Server] Failed to start: {e}", stacklevel=2)

    # Eval-only mode
    if bool(getattr(args, "eval_only_from_run", False)):
        if not os.path.exists(args.temp_easy_root):
            train_init(args)
        final_path = Path(args.results_root) / "final_result.json"
        if not final_path.exists():
            raise FileNotFoundError(f"final_result.json not found: {final_path}")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        checkpoint_state = _load_checkpoint(paths["checkpoint_dir"]) or {}
        extra_params = {str(k): v for k, v in checkpoint_state.get("extra_params", {}).items()}
        _run_eval_stage(args, final, extra_params, paths)
        return

    data_dir = args.data_dir
    data_num = sum(1 for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)))
    if args.data_parallel_size > data_num:
        warnings.warn(f"data_parallel_size ({args.data_parallel_size}) > data_num ({data_num}), clamping.", stacklevel=2)
        args.data_parallel_size = data_num

    execution_worker = ExecutionWorker()
    answers: dict = {}
    extra_params: dict = {}
    count = 0
    best_result: dict = {}
    start_iter = 0
    results = {"time": {}, "prompt": {}, "PAR-2": {}}

    # Resume from checkpoint
    if bool(getattr(args, "resume_from_checkpoint", True)):
        state = _load_checkpoint(paths["checkpoint_dir"])
        if state is not None:
            start_iter = int(state.get("next_iter", 0))
            results = state.get("results", results)
            answers = state.get("answers", answers)
            extra_params = state.get("extra_params", extra_params)
            best_result = state.get("best_result", best_result)
            print(f"[Checkpoint] Resumed from iteration {start_iter}.", flush=True)

    # Baseline
    baseline_initialized = False
    if "0" not in results.get("time", {}):
        train_init(args)
        baseline_cpp = Path(args.temp_root) / "EasySAT_0" / "EasySAT.cpp"
        baseline_cpp.parent.mkdir(parents=True, exist_ok=True)
        rendered = build_solver_source({}, timeout=args.timeout, data_dir=f'"{data_dir}"')
        baseline_cpp.write_text(rendered, encoding="utf-8")

        if args.original:
            success = execution_worker.execute_original(0, args.data_parallel_size)
            filenames = [f"0_{n}.txt" for n in range(args.data_parallel_size)]
            t0 = time.time()
            while True:
                if time.time() - t0 > args.timeout * (2 * data_num / args.data_parallel_size):
                    raise TimeoutError("Baseline solver timed out")
                if all((paths["temp_results_dir"] / f"finished{f}").exists() for f in filenames):
                    results, best_result = collect_results(answers={0: ""}, repetition_dict={}, results={}, args=args)
                    break
        else:
            results["time"]["0"] = args.original_result["time"]
            results["prompt"]["0"] = ""
            results["PAR-2"]["0"] = args.original_result["PAR-2"]
        baseline_initialized = True

    if baseline_initialized:
        print(f"[Baseline] PAR-2={results['PAR-2']['0']} time={results['time']['0']}", flush=True)
        _atomic_write_json(
            paths["results_root"] / "baseline_result.json",
            {"time": results["time"].get("0"), "PAR-2": results["PAR-2"].get("0")},
        )

    baseline_text = SOLVER_BASELINE_CPP.read_text(encoding="utf-8")
    baseline_par2 = results["PAR-2"].get("0")

    # Write-back tracking
    _template_strategy = str(getattr(args, "template_update_strategy", "none") or "none").lower()
    _writeback_par2: dict[str, float] = {}
    _best_codes: dict[str, str] = {}  # best known code per task (for compound injection)
    _sa_rng = random.Random(rand_seed)

    iterations_log: list[dict] = []
    result: dict = {}

    loop_end = int(args.iteration_num)

    for i in range(start_iter, loop_end):
        current_task = _task_for_iteration(task_sequence, selection_mode, i, rand_seed)
        is_all_mode = (current_task == ALL_TASKS_SENTINEL)

        # For args.task use the first task in sequence as a label (needed by evaluate())
        args.task = task_sequence[0] if is_all_mode else current_task
        clean_files(folder_path=args.temp_results_dir, mode="all")

        # Baseline code: dict for all-mode, string for single-task
        if is_all_mode:
            baseline_code: str | dict = {
                t: extract_baseline_section(baseline_text, t) for t in task_sequence
            }
        else:
            baseline_code = extract_baseline_section(baseline_text, current_task)

        # Last-iteration result for feedback prompt
        last_iter_result: dict | None = None
        if i > 0 and best_result:
            bk = list(best_result.keys())[0]
            bd = best_result[bk]
            meta = extra_params.get(int(bk) if str(bk).isdigit() else bk, {})
            if is_all_mode and "implementations" in meta:
                last_iter_result = {"par2": bd[2], "implementations": meta["implementations"]}
            else:
                last_iter_result = {
                    "par2": bd[2],
                    "code": bd[1],
                    "title": meta.get("title", ""),
                    "reason": meta.get("reason", ""),
                }

        prompt_mode = "original" if (i == 0 or not last_iter_result) else "feedback"
        if i > 0 and last_iter_result and check_reIteration(
            round=i,
            best_result_dict=best_result,
            baseline={"time": results["time"]["0"], "PAR-2": results["PAR-2"]["0"]},
        ):
            prompt_mode = "original"

        task_label = "all" if is_all_mode else current_task
        prompt_file = _render_prompt(
            mode=prompt_mode,
            task_name=task_sequence[0] if is_all_mode else current_task,
            baseline_par2=baseline_par2,
            baseline_code=baseline_code,
            last_iter_result=last_iter_result,
            use_structured=getattr(args, "use_structured", False),
            config_payload=getattr(args, "_loaded_config_payload", {}),
            output_path=paths["temp_prompts_dir"] / f"iter_{i}_{prompt_mode}_prompt.txt",
            all_tasks_mode=is_all_mode,
            task_sequence=task_sequence,
        )
        print(f"[Iter {i}] task={task_label} mode={prompt_mode}", flush=True)

        # ── Query LLM (batch) ────────────────────────────────────────────────
        # all-mode: returns {task: {code,title,reason}} per batch slot
        # single-mode: returns {code,title,reason} per batch slot
        answer_payloads: dict[int, dict] = {}   # batch_id → payload
        for batch_id in range(args.batch_size):
            c = i * args.batch_size + batch_id + 1
            if is_all_mode:
                c_out, multi_payload = _query_llm_all(prompt_file, c, args, task_sequence)
                answer_payloads[batch_id] = multi_payload  # {task: {code,title,reason}}
            else:
                c_out, payload = _query_llm(prompt_file, c, args)
                answer_payloads[batch_id] = payload        # {code,title,reason}
            print(f"  [LLM] count={c} all_mode={is_all_mode}", flush=True)

        # ── Execute (compile + run) ──────────────────────────────────────────
        id_list: list[int] = []
        repetition_dict: dict = {}
        for batch_id in range(args.batch_size):
            c = i * args.batch_size + batch_id + 1
            payload = answer_payloads[batch_id]

            if is_all_mode:
                c_out, success, combined_code = _execute_candidate_all(
                    c, results, args, payload, task_sequence
                )
                answers[c_out] = combined_code
                # Store per-task titles/reasons + implementations list for feedback
                impls = [
                    {"task": t, "code": payload[t]["code"],
                     "title": payload[t]["title"], "reason": payload[t]["reason"]}
                    for t in task_sequence if t in payload
                ]
                extra_params[c_out] = {"implementations": impls, "title": "", "reason": ""}
            else:
                c_out, success, code = _execute_candidate(c, results, args, payload, _best_codes)
                answers[c_out] = code
                extra_params[c_out] = {
                    "title": payload.get("title", ""),
                    "reason": payload.get("reason", ""),
                }

            if success:
                id_list.append(c_out)
            elif args.devoid_duplication and not success:
                repetition_dict[c_out] = answers[c_out]

        # ── Wait for solver results ──────────────────────────────────────────
        filenames = [f"{gid}_{n}.txt" for gid in id_list for n in range(args.data_parallel_size)]
        t0 = time.time()
        timeout_limit = args.timeout * (2 * data_num / args.data_parallel_size)
        while True:
            elapsed = time.time() - t0
            if elapsed > timeout_limit:
                warnings.warn(f"[Iter {i}] Solver timeout after {elapsed:.1f}s", stacklevel=2)
                result, best_result = collect_results(
                    answers=answers, repetition_dict=repetition_dict, results=results, args=args)
                delete_InfiniteLoopInst(
                    candidates=[f"finished{f}" for f in filenames], result_dict=result)
                break
            if all((paths["temp_results_dir"] / f"finished{f}").exists() for f in filenames):
                result, best_result = collect_results(
                    answers=answers, repetition_dict=repetition_dict, results=results, args=args)
                break

        if not id_list:
            warnings.warn(f"[Iter {i}] No candidates compiled successfully.", stacklevel=2)
            result = {"time": {}, "prompt": {}, "PAR-2": {}}

        results["time"].update(result.get("time", {}))
        results["PAR-2"].update(result.get("PAR-2", {}))
        results["prompt"].update(result.get("prompt", {}))

        # ── Progress logging ─────────────────────────────────────────────────
        iter_par2 = result.get("PAR-2", {})
        iter_best_par2 = min(iter_par2.values()) if iter_par2 else None
        iter_best_id   = min(iter_par2, key=iter_par2.get) if iter_par2 else None
        iter_meta = extra_params.get(
            int(iter_best_id) if iter_best_id and str(iter_best_id).isdigit() else 0, {}
        )
        # For all-mode dashboard entry: show first implementation as representative
        first_impl = (iter_meta.get("implementations") or [{}])[0]
        iterations_log.append({
            "iter": i,
            "task": task_label,
            "best_par2": iter_best_par2,
            "best_code": (
                first_impl.get("code", "")
                if is_all_mode
                else answers.get(int(iter_best_id) if iter_best_id and str(iter_best_id).isdigit() else 0, "")
            ),
            "title":  first_impl.get("title",  "") if is_all_mode else iter_meta.get("title", ""),
            "reason": first_impl.get("reason", "") if is_all_mode else iter_meta.get("reason", ""),
            "implementations": iter_meta.get("implementations", []) if is_all_mode else [],
        })
        _update_progress(args, run_id, baseline_par2, iterations_log)

        _save_iteration_artifacts(
            i, result, best_result,
            paths["temp_prompts_dir"], paths["results_root"], paths["snapshots_dir"],
        )
        _save_checkpoint(
            i + 1, results, answers, extra_params, best_result, paths["checkpoint_dir"], run_id
        )

        # ── Optional write-back ──────────────────────────────────────────────
        if _template_strategy in ("greedy_train", "annealing") and best_result:
            bk = list(best_result.keys())[0]
            bp2 = best_result[bk][2]
            bc  = best_result[bk][1]
            if bp2 < (baseline_par2 or float("inf")):
                if is_all_mode:
                    # bc is a JSON string of {task: {code,...}}; update all tasks
                    try:
                        multi = json.loads(bc)
                        wb_codes = {
                            t: info["code"]
                            for t, info in multi.items()
                            if isinstance(info, dict) and len(str(info.get("code", "")).strip()) >= 10
                        }
                        if wb_codes:
                            # Use the minimum prev to decide accept/reject
                            prev = min(_writeback_par2.get(t, float("inf")) for t in wb_codes)
                            accept, reason_str = _should_writeback(
                                _template_strategy, bp2, prev, baseline_par2 or 1.0, i, loop_end, _sa_rng
                            )
                            print(f"[WriteBack] iter={i} all-mode PAR-2={bp2:.0f} → {reason_str}", flush=True)
                            if accept:
                                _best_codes.update(wb_codes)
                                for t in wb_codes:
                                    _writeback_par2[t] = bp2
                                update_solver_template(_best_codes)
                    except (json.JSONDecodeError, AttributeError):
                        pass
                elif len(str(bc).strip()) >= 10:
                    task_wb = _infer_task_from_code(bc) or current_task
                    prev = _writeback_par2.get(task_wb, float("inf"))
                    accept, reason_str = _should_writeback(
                        _template_strategy, bp2, prev, baseline_par2 or 1.0, i, loop_end, _sa_rng
                    )
                    print(f"[WriteBack] iter={i} task={task_wb} PAR-2={bp2:.0f} → {reason_str}", flush=True)
                    if accept:
                        _best_codes[task_wb] = bc
                        _writeback_par2[task_wb] = bp2
                        update_solver_template(_best_codes)

    # Save final result
    final = {
        k: {"time": results["time"][k], "PAR-2": results["PAR-2"][k], "prompt": results["prompt"].get(k, "")}
        for k in results["time"]
    }
    for p in (paths["temp_prompts_dir"] / "final_result.json", paths["results_root"] / "final_result.json"):
        _atomic_write_json(p, final)

    # Training curve plot
    try:
        plot_training_curve(results_root=paths["results_root"], baseline_par2=baseline_par2, run_id=run_id)
    except Exception as exc:
        warnings.warn(f"[Plotting] {exc}", stacklevel=2)

    # Eval stage
    if not bool(getattr(args, "run_eval", True)):
        print("[Eval] Skipped (run_eval=False)", flush=True)
        return
    _run_eval_stage(args, final, {str(k): v for k, v in extra_params.items()}, paths)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")

    parser.add_argument("--iteration_num", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_parallel_size", type=int, default=2)
    parser.add_argument("--devoid_duplication", type=bool, default=False)
    parser.add_argument("--llm_model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--timeout", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./temp/data_train")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task", type=str, default="bump_var_function")
    parser.add_argument("--optimize_tasks", nargs="*", default=None)
    parser.add_argument("--task_selection_mode", type=str, default="random_one")
    parser.add_argument("--original", type=bool, default=True)
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=True)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--run_eval", type=bool, default=True)
    parser.add_argument("--eval_only_from_run", type=bool, default=False)
    parser.add_argument("--template_update_strategy", type=str, default="none")
    parser.add_argument("--enable_server", type=bool, default=False)
    parser.add_argument("--server_port", type=int, default=8080)
    parser.add_argument("--use_structured", type=bool, default=True)

    args = parser.parse_args()

    loaded_config: dict = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            loaded_config = yaml.safe_load(f) or {}
        for k, v in loaded_config.items():
            setattr(args, k, v)

    args = _apply_env_overrides(args)
    args._loaded_config_payload = loaded_config

    def _sig(signum, frame):
        _graceful_shutdown(f"Signal {signum}", exit_code=128 + int(signum))

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        main(args)
    except KeyboardInterrupt:
        _graceful_shutdown("KeyboardInterrupt", exit_code=130)
    finally:
        _graceful_shutdown("Exit")
