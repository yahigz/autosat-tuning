"""Prompt building for reusable solver tuning."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from jinja2 import Template

from .tasks import TaskSpec, task_specs_from_config


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
    default_modules: Mapping[str, Mapping[str, str]] | None = None,
) -> List[Dict[str, str]]:
    config_payload = dict(config_payload or {})
    module_specs: Dict[str, Dict[str, str]] = {
        name: dict(spec) for name, spec in (default_modules or DEFAULT_HEURISTIC_MODULES).items()
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
        lines.append(f"=== {name} - {m.get('label', '')} ===")
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


def _normalize_structured_fields(fields: Sequence[str] | None) -> list[str]:
    normalized = [str(field).strip() for field in (fields or ("code", "title", "reason")) if str(field).strip()]
    return normalized or ["code", "title", "reason"]


def _format_structured_field_block(fields: Sequence[str]) -> str:
    lines = ["Return a valid JSON object with exactly these fields:"]
    for field in fields:
        lines.append(f'  "{field}": string')
    lines.append("All fields are required.")
    return "\n".join(lines)


def _task_to_mapping(task: TaskSpec | Mapping[str, Any]) -> Dict[str, str]:
    if isinstance(task, TaskSpec):
        return {
            "name": task.name,
            "label": task.label,
            "description": task.description,
            "signature": task.signature,
            "insertion_format": task.insertion_format,
            "baseline_code": task.baseline_code,
        }
    return {
        "name": str(task.get("name", "")),
        "label": str(task.get("label", "")),
        "description": str(task.get("description", "")),
        "signature": str(task.get("signature", "")),
        "insertion_format": str(task.get("insertion_format", "")),
        "baseline_code": str(task.get("baseline_code", "")),
    }


def _format_baseline_section(
    task_name: str,
    baseline_par2: float | None,
    baseline_code: str | Mapping[str, str] = "",
) -> str:
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
    lines = ["=== Last iteration result ==="]
    par2 = last_iter.get("par2")
    if par2 is not None:
        lines.append(f"PAR-2: {par2}")

    impls = last_iter.get("implementations")
    if impls:
        for impl in impls:
            t = impl.get("task", "")
            title = str(impl.get("title", "") or "").strip()
            reason = str(impl.get("reason", "") or "").strip()
            code = str(impl.get("code", "") or "").strip()
            lines.append(f"\n--- {t} ---")
            if title:
                lines.append(f"Title: {title}")
            if reason:
                lines.append(f"Reason: {reason}")
            if code:
                lines.append(f"Code:\n{code}")
    else:
        title = str(last_iter.get("title", "") or "").strip()
        reason = str(last_iter.get("reason", "") or "").strip()
        code = str(last_iter.get("code", "") or "").strip()
        if title:
            lines.append(f"Title: {title}")
        if reason:
            lines.append(f"Reason: {reason}")
        if code:
            lines.append(f"Code:\n{code}")
    return "\n".join(lines)


def _format_structured_section_single(task_name: str, fields: Sequence[str] | None = None) -> str:
    fields = _normalize_structured_fields(fields)
    if fields == ["code", "title", "reason"]:
        return (
            "Return a valid JSON object with exactly these fields:\n"
            '  "code": string - complete C++ implementation to inject\n'
            '  "title": string - short name for the proposed change\n'
            '  "reason": string - brief motivation for the change, it is advisable to explain the benefit in comparison to the previous iteration or baseline\n'
            "All fields are required."
        )
    return _format_structured_field_block(fields)


def _format_structured_section_all(tasks: Sequence[str], fields: Sequence[str] | None = None) -> str:
    fields = _normalize_structured_fields(fields)
    task_list = ", ".join(f'"{t}"' for t in tasks)
    inner = ", ".join(f'"{field}": "<{field}>"' for field in fields)
    return (
        "Return a valid JSON object with this structure:\n"
        "{\n"
        '  "implementations": [\n'
        f"    {{{{\"task\": \"<name>\", {inner}}}}},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        f"Include one entry for each of: {task_list}.\n"
        "All fields in each entry are required. "
        "If you have no improvement for a task, still include it with the baseline code."
    )


def build_prompt_text_for_tasks(
    base_prompt_text: str,
    task_specs: Sequence[TaskSpec | Mapping[str, Any]],
    baseline_par2: float | None = None,
    baseline_info: Mapping[str, Any] | None = None,
    baseline_codes: Mapping[str, str] | None = None,
    last_iter_result: Mapping[str, Any] | None = None,
    structured_output: bool = False,
    all_tasks_mode: bool = False,
    default_modules: Mapping[str, Mapping[str, str]] | None = None,
    structured_output_fields: Sequence[str] | None = None,
) -> str:
    modules = [_task_to_mapping(task) for task in task_specs]
    if baseline_codes is None:
        baseline_codes = {task["name"]: task.get("baseline_code", "") for task in modules}

    heuristics_section = _format_heuristics_section(modules, baseline_codes)
    if default_modules:
        # Allow callers to pre-load their project-specific task descriptions.
        task_specs = task_specs_from_config(
            {"tasks": dict(default_modules)},
            allowed_names=[m["name"] for m in modules] if modules else None,
            fallback_names=[m["name"] for m in modules] if modules else None,
        )
        if task_specs:
            modules = [_task_to_mapping(task) for task in task_specs]
            heuristics_section = _format_heuristics_section(modules, baseline_codes)

    if baseline_info is not None:
        baseline_section = ["=== Baseline (iteration 0) ==="]
        for key, value in baseline_info.items():
            if value is None:
                continue
            baseline_section.append(f"{key}: {value}")
        if baseline_codes:
            if all_tasks_mode:
                for t, code in baseline_codes.items():
                    if code:
                        baseline_section.append(f"\n{t}:\n{code}")
            elif modules:
                code = baseline_codes.get(modules[0]["name"], "")
                baseline_section.append(f"Task: {modules[0]['name']}")
                if code:
                    baseline_section.append(f"Code:\n{code}")
        baseline_section = "\n".join(baseline_section)
    else:
        baseline_section = (
            _format_baseline_section(
                task_name="all" if all_tasks_mode else (modules[0]["name"] if modules else ""),
                baseline_par2=baseline_par2,
                baseline_code=baseline_codes if all_tasks_mode else (baseline_codes.get(modules[0]["name"], "") if modules else ""),
            )
            if (baseline_par2 is not None or baseline_codes)
            else ""
        )
    last_iter_section = _format_last_iter_section(last_iter_result) if last_iter_result else ""

    rendered = Template(base_prompt_text).render(
        heuristics_section=heuristics_section,
        baseline_section=baseline_section,
        last_iter_section=last_iter_section,
    )

    if structured_output:
        if all_tasks_mode:
            rendered = rendered.rstrip() + "\n\n" + _format_structured_section_all([task["name"] for task in modules], fields=structured_output_fields)
        else:
            rendered = rendered.rstrip() + "\n\n" + _format_structured_section_single(modules[0]["name"] if modules else "", fields=structured_output_fields)

    return rendered


def build_prompt_text(
    base_prompt_text: str,
    task_name: str,
    baseline_par2: float | None = None,
    baseline_info: Mapping[str, Any] | None = None,
    baseline_code: str | Mapping[str, str] = "",
    last_iter_result: Mapping[str, Any] | None = None,
    config_payload: Mapping[str, Any] | None = None,
    structured_output: bool = False,
    allowed_modules: Sequence[str] | None = None,
    all_tasks_mode: bool = False,
    default_modules: Mapping[str, Mapping[str, str]] | None = None,
    structured_output_fields: Sequence[str] | None = None,
) -> str:
    task_specs = task_specs_from_config(
        config_payload or {},
        allowed_names=allowed_modules if allowed_modules else ([task_name] if not all_tasks_mode else None),
        baseline_codes=baseline_code if isinstance(baseline_code, Mapping) else {task_name: baseline_code} if baseline_code else {},
        fallback_names=[task_name] if task_name else None,
        default_specs=default_modules,
    )
    return build_prompt_text_for_tasks(
        base_prompt_text=base_prompt_text,
        task_specs=task_specs,
        baseline_par2=baseline_par2,
        baseline_info=baseline_info,
        baseline_codes=baseline_code if isinstance(baseline_code, Mapping) else {task_name: baseline_code} if baseline_code else {},
        last_iter_result=last_iter_result,
        structured_output=structured_output,
        all_tasks_mode=all_tasks_mode,
        default_modules=default_modules,
        structured_output_fields=structured_output_fields,
    )


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
    empty = {t: {"code": "", "title": "", "reason": ""} for t in tasks}
    try:
        data = json.loads(content)
    except Exception as exc:
        print(f"[StructuredOutput][all] JSON parse error: {exc}. Raw: {content[:300]}", flush=True)
        return empty

    impls = data.get("implementations", [])
    if not isinstance(impls, list):
        print("[StructuredOutput][all] 'implementations' is not a list.", flush=True)
        return empty

    result: Dict[str, Dict[str, str]] = {}
    for item in impls:
        if not isinstance(item, dict):
            continue
        t = str(item.get("task", "") or "").strip()
        if not t:
            continue
        result[t] = {
            "code": str(item.get("code", "") or "").strip(),
            "title": str(item.get("title", "") or "").strip(),
            "reason": str(item.get("reason", "") or "").strip(),
        }

    for t in tasks:
        if t not in result:
            result[t] = {"code": "", "title": "", "reason": ""}
    return result


def get_code_from_text_all(raw_text: str, tasks: Sequence[str]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for t in tasks:
        pattern = re.compile(
            r"//\s*start_" + re.escape(t) + r"\s*\n(.*?)\n//\s*end_" + re.escape(t),
            re.DOTALL,
        )
        m = pattern.search(raw_text)
        result[t] = {"code": m.group(1).strip() if m else "", "title": "", "reason": ""}
    return result


def build_tool_schema(
    config_payload: Mapping[str, Any] | None = None,
    task_name: str | None = None,
    fields: Sequence[str] | None = None,
) -> Dict[str, Any]:
    fields = _normalize_structured_fields(fields)
    return {
        "name": "submit_heuristic_code",
        "description": "Submit the improved heuristic implementation.",
        "input_schema": {
            "type": "object",
            "properties": {
                field: {"type": "string", "description": field.replace("_", " ").title()}
                for field in fields
            },
            "required": list(fields),
        },
    }


def build_tool_schema_all(tasks: Sequence[str], fields: Sequence[str] | None = None) -> Dict[str, Any]:
    fields = _normalize_structured_fields(fields)
    item_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": f"One of: {', '.join(tasks)}"},
            **{
                field: {"type": "string", "description": field.replace("_", " ").title()}
                for field in fields
            },
        },
        "required": ["task", *fields],
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


def build_structured_schema_all(tasks: Sequence[str], fields: Sequence[str] | None = None) -> Dict[str, Any]:
    fields = _normalize_structured_fields(fields)
    item_schema = {
        "type": "object",
        "properties": {"task": {"type": "string"}, **{field: {"type": "string"} for field in fields}},
        "required": ["task", *fields],
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


def build_structured_schema(
    config_payload: Mapping[str, Any] | None = None,
    task_name: str | None = None,
    fields: Sequence[str] | None = None,
) -> Dict[str, Any]:
    fields = _normalize_structured_fields(fields)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "heuristic_code_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    field: {"type": "string", "description": field.replace("_", " ").title()}
                    for field in fields
                },
                "required": list(fields),
                "additionalProperties": False,
            },
        },
    }
