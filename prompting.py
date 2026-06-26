"""Prompt building for AutoSAT tuning."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from jinja2 import Template


DEFAULT_HEURISTIC_MODULES: Dict[str, Dict[str, str]] = {
    "bump_var_function": {
        "label": "Variable activity bumping",
        "description": (
            "This function increases the priority of variables that cause recent conflicts; as a result, it helps the solver focus its next choices on the most difficult parts of the problem."
        ),
        "insertion_format": "Replace the full function body inside `void Solver::bump_var(int var, double coeff)`.",
        "signature": "void Solver::bump_var(int var, double coeff)",
    },
    "restart_condition": {
        "label": "Restart trigger",
        "description": (
            "This is a simple predicate that returns true or false to decide if the current search branch should be stopped; its main purpose is to prevent the solver from wasting too much time on a single, dead-end path."
        ),
        "insertion_format": (
            "Replace the `else if (...)` branch that calls `restart()` inside `solve()`. "
            "Keep valid C++ `else if` syntax."
        ),
        "signature": "else if ( ... ) restart();",
    },
    "restart_function": {
        "label": "Restart policy",
        "description": (
            "This cancels the current temporary choices and returns the search to the very beginning while keeping all learned rules; applying it allows the solver to escape trapped areas and try better paths."
        ),
        "insertion_format": "Replace the full function body inside `void Solver::restart()`.",
        "signature": "void Solver::restart()",
    },
    "rephase_function": {
        "label": "Rephase policy",
        "description": (
            "This changes the preferred true/false values of variables according to a specific plan; by doing this, it forces the solver to explore completely different areas of the formula."
        ),
        "insertion_format": "Replace the full function body inside `void Solver::rephase()`.",
        "signature": "void Solver::rephase()",
    },
    "rephase_condition": {
        "label": "Rephase trigger",
        "description": (
            "This is a simple predicate that returns true or false to signal when it is time to change the true/false choice strategy; this stops the solver from repeating the same phase mistakes over and over."
        ),
        "insertion_format": (
            "Replace the `else if (...)` branch that calls `rephase()` inside `solve()`. "
            "Keep valid C++ `else if` syntax."
        ),
        "signature": "else if ( ... ) rephase();",
    },
}


def _normalize(v: Any) -> str:
    return str(v or "").strip().strip("/")


def load_heuristic_modules(
    config_payload: Mapping[str, Any] | None,
    allowed_modules: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    config_payload = dict(config_payload or {})
    module_specs: Dict[str, Dict[str, str]] = {
        name: dict(spec) for name, spec in DEFAULT_HEURISTIC_MODULES.items()
    }

    overrides = config_payload.get("heuristic_modules")
    if isinstance(overrides, Mapping):
        for name, spec in overrides.items():
            normalized = _normalize(name)
            base = dict(module_specs.get(normalized, {}))
            if isinstance(spec, Mapping):
                for k, v in spec.items():
                    if v is not None:
                        base[str(k)] = v
            module_specs[normalized] = base

    order: List[str] = []
    for key in ("heuristic_module_order", "optimize_tasks"):
        raw = config_payload.get(key)
        if raw:
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            order.extend(_normalize(x) for x in items)
            break
    if not order and config_payload.get("task"):
        order.append(_normalize(config_payload["task"]))
    if not order:
        order.extend(module_specs.keys())

    allowed = {_normalize(x) for x in allowed_modules} if allowed_modules else None
    seen: set = set()
    result: List[Dict[str, str]] = []
    for name in order:
        if not name or name in seen:
            continue
        if allowed is not None and name not in allowed:
            continue
        spec = module_specs.get(name)
        if not spec:
            continue
        result.append({
            "name": name,
            "label": str(spec.get("label", name.replace("_", " "))),
            "description": str(spec.get("description", "")),
            "insertion_format": str(spec.get("insertion_format", "")),
            "signature": str(spec.get("signature", name)),
        })
        seen.add(name)
    return result


def _format_heuristics_section(
    modules: Sequence[Mapping[str, Any]],
    baseline_codes: Mapping[str, str] | None = None,
) -> str:
    if not modules:
        return ""
    lines: List[str] = []
    for m in modules:
        name = m.get("name", "")
        lines.append(f"=== {name} — {m.get('label', '')} ===")
        desc = str(m.get("description", "")).strip()
        if desc:
            lines.append(f"Description: {desc}")
        sig = str(m.get("signature", "")).strip()
        fmt = str(m.get("insertion_format", "")).strip()
        if sig:
            lines.append(f"Signature: {sig}")
        if fmt:
            lines.append(f"Insertion: {fmt}")
        code = (baseline_codes or {}).get(name, "")
        if code:
            lines.append(f"Baseline code:\n{code}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_baseline_section(
    task_name: str,
    baseline_par2: float | None,
    baseline_code: str | Mapping[str, str] = "",
) -> str:
    """Format baseline block for single task or all tasks (baseline_code is dict in 'all' mode)."""
    lines = ["=== Baseline (iteration 0) ==="]
    if baseline_par2 is not None:
        lines.append(f"PAR-2: {baseline_par2}")
    if isinstance(baseline_code, Mapping):
        for t, c in baseline_code.items():
            if c:
                lines.append(f"\n{t}:\n{c}")
    else:
        lines.append(f"Task: {task_name}")
        if baseline_code:
            lines.append(f"Code:\n{baseline_code}")
    return "\n".join(lines)


def _format_last_iter_section(last_iter: Mapping[str, Any]) -> str:
    """Format last-iteration block for single task or all tasks.

    Single-task shape:  {par2, code, title, reason}
    All-tasks shape:    {par2, implementations: [{task, code, title, reason}, ...]}
    """
    lines = ["=== Last iteration result ==="]
    par2 = last_iter.get("par2")
    if par2 is not None:
        lines.append(f"PAR-2: {par2}")

    impls = last_iter.get("implementations")
    if impls:
        for impl in impls:
            t = impl.get("task", "")
            title  = str(impl.get("title",  "") or "").strip()
            reason = str(impl.get("reason", "") or "").strip()
            code   = str(impl.get("code",   "") or "").strip()
            lines.append(f"\n--- {t} ---")
            if title:  lines.append(f"Title: {title}")
            if reason: lines.append(f"Reason: {reason}")
            if code:   lines.append(f"Code:\n{code}")
    else:
        title  = str(last_iter.get("title",  "") or "").strip()
        reason = str(last_iter.get("reason", "") or "").strip()
        code   = str(last_iter.get("code",   "") or "").strip()
        if title:  lines.append(f"Title: {title}")
        if reason: lines.append(f"Reason: {reason}")
        if code:   lines.append(f"Code:\n{code}")
    return "\n".join(lines)


def _format_structured_section_single(task_name: str) -> str:
    return (
        "Return a valid JSON object with exactly these fields:\n"
        '  "code": string — complete C++ implementation to inject\n'
        '  "title": string — short name for the proposed change\n'
        '  "reason": string — brief motivation for the change\n'
        "All fields are required."
    )


def _format_structured_section_all(tasks: Sequence[str]) -> str:
    task_list = ", ".join(f'"{t}"' for t in tasks)
    return (
        "Return a valid JSON object with this structure:\n"
        '{\n'
        '  "implementations": [\n'
        '    {"task": "<name>", "code": "<C++ code>", "title": "<short name>", "reason": "<motivation>"},\n'
        '    ...\n'
        '  ]\n'
        '}\n'
        f'Include one entry for each of: {task_list}.\n'
        "All fields in each entry are required. "
        "If you have no improvement for a task, still include it with the baseline code."
    )


def build_prompt_text(
    base_prompt_text: str,
    task_name: str,
    baseline_par2: float | None = None,
    baseline_code: str | Mapping[str, str] = "",
    last_iter_result: Mapping[str, Any] | None = None,
    config_payload: Mapping[str, Any] | None = None,
    structured_output: bool = False,
    allowed_modules: Sequence[str] | None = None,
    all_tasks_mode: bool = False,
) -> str:
    modules = load_heuristic_modules(
        config_payload or {},
        allowed_modules=allowed_modules if allowed_modules else ([task_name] if not all_tasks_mode else None),
    )
    baseline_codes: Mapping[str, str] = (
        baseline_code if isinstance(baseline_code, Mapping)
        else ({task_name: baseline_code} if baseline_code else {})
    )

    heuristics_section = _format_heuristics_section(modules, baseline_codes)
    baseline_section = (
        _format_baseline_section(task_name, baseline_par2, baseline_code)
        if (baseline_par2 is not None or baseline_code)
        else ""
    )
    last_iter_section = _format_last_iter_section(last_iter_result) if last_iter_result else ""

    context = {
        "heuristics_section": heuristics_section,
        "baseline_section": baseline_section,
        "last_iter_section": last_iter_section,
    }
    rendered = Template(base_prompt_text).render(**context)

    if structured_output:
        if all_tasks_mode:
            task_names = [m["name"] for m in modules]
            rendered = rendered.rstrip() + "\n\n" + _format_structured_section_all(task_names)
        else:
            rendered = rendered.rstrip() + "\n\n" + _format_structured_section_single(task_name)

    return rendered


def load_prompt_text(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8")


def parse_structured_response(content: str) -> Dict[str, str]:
    try:
        data = json.loads(content)
    except Exception as exc:
        print(f"[StructuredOutput] JSON parse error: {exc}. Raw: {content[:300]}", flush=True)
        return {"code": "", "title": "", "reason": ""}
    return {
        "code": str(data.get("code", "") or "").strip(),
        "title": str(data.get("title", "") or "").strip(),
        "reason": str(data.get("reason", "") or "").strip(),
    }


def parse_multi_response(content: str, tasks: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Parse a multi-task structured response.

    Expected JSON shape:
      {"implementations": [{"task": "...", "code": "...", "title": "...", "reason": "..."}, ...]}

    Returns: {task_name: {code, title, reason}}
    Missing or invalid tasks get empty strings.
    """
    empty = {t: {"code": "", "title": "", "reason": ""} for t in tasks}
    try:
        data = json.loads(content)
    except Exception as exc:
        print(f"[StructuredOutput][all] JSON parse error: {exc}. Raw: {content[:300]}", flush=True)
        return empty

    impls = data.get("implementations", [])
    if not isinstance(impls, list):
        print(f"[StructuredOutput][all] 'implementations' is not a list.", flush=True)
        return empty

    result: Dict[str, Dict[str, str]] = {}
    for item in impls:
        if not isinstance(item, dict):
            continue
        t = str(item.get("task", "") or "").strip()
        if not t:
            continue
        result[t] = {
            "code":   str(item.get("code",   "") or "").strip(),
            "title":  str(item.get("title",  "") or "").strip(),
            "reason": str(item.get("reason", "") or "").strip(),
        }

    for t in tasks:
        if t not in result:
            result[t] = {"code": "", "title": "", "reason": ""}
    return result


def get_code_from_text_all(raw_text: str, tasks: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Extract per-task code from plain text using // start_TASK / // end_TASK markers."""
    result: Dict[str, Dict[str, str]] = {}
    for t in tasks:
        pattern = re.compile(
            r"//\s*start_" + re.escape(t) + r"\s*\n(.*?)\n//\s*end_" + re.escape(t),
            re.DOTALL,
        )
        m = pattern.search(raw_text)
        result[t] = {"code": m.group(1).strip() if m else "", "title": "", "reason": ""}
    return result


def build_tool_schema(config_payload: Mapping[str, Any] | None = None, task_name: str | None = None) -> Dict[str, Any]:
    """Anthropic tool_use schema for Eliza structured output."""
    return {
        "name": "submit_heuristic_code",
        "description": "Submit the improved heuristic implementation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code":   {"type": "string", "description": "Complete C++ implementation to inject."},
                "title":  {"type": "string", "description": "Short name for the proposed change."},
                "reason": {"type": "string", "description": "Brief motivation for the change."},
            },
            "required": ["code", "title", "reason"],
        },
    }


def build_tool_schema_all(tasks: Sequence[str]) -> Dict[str, Any]:
    """Anthropic tool_use schema for all-tasks mode (Eliza)."""
    item_schema = {
        "type": "object",
        "properties": {
            "task":   {"type": "string", "description": f"One of: {', '.join(tasks)}"},
            "code":   {"type": "string", "description": "Complete C++ implementation."},
            "title":  {"type": "string", "description": "Short change name."},
            "reason": {"type": "string", "description": "Brief motivation."},
        },
        "required": ["task", "code", "title", "reason"],
    }
    return {
        "name": "submit_all_heuristics",
        "description": "Submit improved implementations for all heuristic functions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "implementations": {
                    "type": "array",
                    "items": item_schema,
                    "description": f"One entry per task: {', '.join(tasks)}",
                }
            },
            "required": ["implementations"],
        },
    }


def build_structured_schema_all(tasks: Sequence[str]) -> Dict[str, Any]:
    """OpenAI json_schema for all-tasks mode."""
    item_schema = {
        "type": "object",
        "properties": {
            "task":   {"type": "string"},
            "code":   {"type": "string"},
            "title":  {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["task", "code", "title", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "heuristic_code_response_all",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "implementations": {
                        "type": "array",
                        "items": item_schema,
                    }
                },
                "required": ["implementations"],
                "additionalProperties": False,
            },
        },
    }


def build_structured_schema(config_payload: Mapping[str, Any] | None = None, task_name: str | None = None) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "heuristic_code_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "code":   {"type": "string", "description": "Complete C++ implementation to inject."},
                    "title":  {"type": "string", "description": "Short name for the proposed change."},
                    "reason": {"type": "string", "description": "Brief motivation for the change."},
                },
                "required": ["code", "title", "reason"],
                "additionalProperties": False,
            },
        },
    }
