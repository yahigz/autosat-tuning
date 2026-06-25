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
            "Increases the priority of variables involved in recent conflicts, "
            "helping the solver focus on the hardest parts of the problem."
        ),
        "insertion_format": "Replace the full function body inside `void Solver::bump_var(int var, double coeff)`.",
        "signature": "void Solver::bump_var(int var, double coeff)",
    },
    "restart_condition": {
        "label": "Restart trigger",
        "description": (
            "Predicate deciding whether to restart the current search. "
            "Prevents wasting time on dead-end search paths."
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
            "Cancels current temporary assignments and returns to the root level "
            "while keeping all learned clauses."
        ),
        "insertion_format": "Replace the full function body inside `void Solver::restart()`.",
        "signature": "void Solver::restart()",
    },
    "rephase_function": {
        "label": "Rephase policy",
        "description": (
            "Changes preferred polarities of variables to escape repeated bad polarity patterns."
        ),
        "insertion_format": "Replace the full function body inside `void Solver::rephase()`.",
        "signature": "void Solver::rephase()",
    },
    "rephase_condition": {
        "label": "Rephase trigger",
        "description": (
            "Predicate deciding when to call rephase(). "
            "Stops the solver from repeating the same polarity mistakes."
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


def _format_baseline_section(task_name: str, baseline_par2: float | None, baseline_code: str) -> str:
    lines = ["=== Baseline (iteration 0) ==="]
    if baseline_par2 is not None:
        lines.append(f"PAR-2: {baseline_par2}")
    lines.append(f"Task: {task_name}")
    if baseline_code:
        lines.append(f"Code:\n{baseline_code}")
    return "\n".join(lines)


def _format_last_iter_section(last_iter: Mapping[str, Any]) -> str:
    lines = ["=== Last iteration result ==="]
    par2 = last_iter.get("par2")
    if par2 is not None:
        lines.append(f"PAR-2: {par2}")
    title = str(last_iter.get("title", "")).strip()
    reason = str(last_iter.get("reason", "")).strip()
    code = str(last_iter.get("code", "")).strip()
    if title:
        lines.append(f"Title: {title}")
    if reason:
        lines.append(f"Reason: {reason}")
    if code:
        lines.append(f"Code:\n{code}")
    return "\n".join(lines)


def _format_structured_section(config_payload: Mapping[str, Any] | None, task_name: str) -> str:
    return (
        "Return a valid JSON object with exactly these fields:\n"
        '  "code": string — complete C++ implementation to inject\n'
        '  "title": string — short name for the proposed change\n'
        '  "reason": string — brief motivation for the change\n'
        "All fields are required."
    )


def build_prompt_text(
    base_prompt_text: str,
    task_name: str,
    baseline_par2: float | None = None,
    baseline_code: str = "",
    last_iter_result: Mapping[str, Any] | None = None,
    config_payload: Mapping[str, Any] | None = None,
    structured_output: bool = False,
    allowed_modules: Sequence[str] | None = None,
) -> str:
    modules = load_heuristic_modules(config_payload or {}, allowed_modules=allowed_modules or [task_name])
    baseline_codes = {task_name: baseline_code} if baseline_code else {}

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
        rendered = rendered.rstrip() + "\n\n" + _format_structured_section(config_payload, task_name)

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
