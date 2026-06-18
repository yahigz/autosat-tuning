from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


DEFAULT_HEURISTIC_MODULES: Dict[str, Dict[str, str]] = {
    "bump_var_function": {
        "label": "Variable activity bumping",
        "description": (
            "This function increases the priority of variables that cause recent conflicts; as a result, it helps the solver focus its next choices on the most difficult parts of the problem."
            "Useful for balancing short-term conflict focus and long-term search stability."
        ),
        "insertion_format": (
            "Replace the full function body inside `void Solver::bump_var(int var, double coeff)`. Provide whole function"
        ),
        "signature": "void Solver::bump_var(int var, double coeff)",
    },
    "restart_condition": {
        "label": "Restart trigger",
        "description": (
            "This is a simple predicate that returns true or false to decide if the current search branch should be stopped; its main purpose is to prevent the solver from wasting too much time on a single, dead-end path. "
            "Useful for changing restart frequency and escape pressure."
        ),
        "insertion_format": (
            "Replace the conditional branch that decides whether `restart()` should be called. "
            "Keep the surrounding `else if` structure and preserve valid C++ syntax."
        ),
        "signature": "else if ( ... ) restart();",
    },
    "restart_function": {
        "label": "Restart policy",
        "description": (
            "This cancels the current temporary choices and returns the search to the very beginning while keeping all learned rules; applying it allows the solver to escape trapped areas and try better paths. "
            "Useful for resetting aggressively or preserving useful search memory."
        ),
        "insertion_format": (
            "Replace the full function body inside `void Solver::restart()`. Provide whole function"
        ),
        "signature": "void Solver::restart()",
    },
    "rephase_function": {
        "label": "Rephase policy",
        "description": (
            "This changes the preferred true/false values of variables according to a specific plan; by doing this, it forces the solver to explore completely different areas of the formula. "
            "Useful for escaping repeated bad polarity patterns."
        ),
        "insertion_format": (
            "Replace the full function body inside `void Solver::rephase()`.  Provide whole function"
        ),
        "signature": "void Solver::rephase()",
    },
    "rephase_condition": {
        "label": "Rephase trigger",
        "description": (
            "This is a simple predicate that returns true or false to signal when it is time to change the true/false choice strategy; this stops the solver from repeating the same phase mistakes over and over."
        ),
        "insertion_format": (
             "Replace the conditional branch that decides whether `rephase()` should be called. "
            "Keep the surrounding `else if` structure and preserve valid C++ syntax."
        ),
        "signature": "else if ( ... ) rephase();",
    }
}


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().strip("/")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _as_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def load_heuristic_modules(config_payload: Mapping[str, Any] | None, allowed_modules: Sequence[str] | None = None) -> List[Dict[str, str]]:
    config_payload = dict(config_payload or {})
    configured = config_payload.get("heuristic_modules")
    module_specs: Dict[str, Dict[str, str]] = {name: dict(spec) for name, spec in DEFAULT_HEURISTIC_MODULES.items()}

    if isinstance(configured, Mapping):
        for name, override in configured.items():
            normalized = _normalize_name(name)
            base = dict(module_specs.get(normalized, {}))
            if isinstance(override, Mapping):
                for key, value in override.items():
                    if value is not None:
                        base[str(key)] = value
            module_specs[normalized] = base

    order: List[str] = []
    explicit_order = config_payload.get("heuristic_module_order")
    if explicit_order:
        order.extend(_normalize_name(item) for item in _as_sequence(explicit_order))
    else:
        optimize_tasks = config_payload.get("optimize_tasks")
        if optimize_tasks:
            order.extend(_normalize_name(item) for item in _as_sequence(optimize_tasks))
        elif config_payload.get("task"):
            order.append(_normalize_name(config_payload.get("task")))

    if not order:
        order.extend(module_specs.keys())

    allowed = {_normalize_name(item) for item in allowed_modules} if allowed_modules else None
    selected: List[Dict[str, str]] = []
    seen = set()
    for name in order:
        if not name or name in seen:
            continue
        if allowed is not None and name not in allowed:
            continue
        spec = module_specs.get(name)
        if not spec:
            continue
        selected.append({
            "name": name,
            "label": str(spec.get("label", name.replace("_", " "))),
            "description": str(spec.get("description", "")),
            "insertion_format": str(spec.get("insertion_format", "")),
            "signature": str(spec.get("signature", name)),
        })
        seen.add(name)

    if not selected:
        for name, spec in module_specs.items():
            if allowed is not None and name not in allowed:
                continue
            selected.append({
                "name": name,
                "label": str(spec.get("label", name.replace("_", " "))),
                "description": str(spec.get("description", "")),
                "insertion_format": str(spec.get("insertion_format", "")),
                "signature": str(spec.get("signature", name)),
            })

    return selected


def format_heuristic_modules_section(modules: Sequence[Mapping[str, Any]]) -> str:
    if not modules:
        return ""
    lines = ["Heuristic modules available for optimization:"]
    for idx, module in enumerate(modules, start=1):
        lines.append(f"{idx}. {module.get('name', '')} - {module.get('label', '')}")
        description = str(module.get("description", "")).strip()
        if description:
            lines.append(f"   Description: {description}")
        signature = str(module.get("signature", "")).strip()
        insertion_format = str(module.get("insertion_format", "")).strip()
        if signature or insertion_format:
            lines.append(f"   Signature: {signature}")
            lines.append(f"   Insertion format: {insertion_format}")
    return "\n".join(lines)


def resolve_structured_output_fields(config_payload: Mapping[str, Any] | None, task_name: str | None = None) -> List[Dict[str, Any]]:
    config_payload = dict(config_payload or {})
    task_name = _normalize_name(task_name or config_payload.get("task"))

    field_specs: List[Dict[str, Any]] = [
        {
            "name": "code",
            "type": "string",
            "description": "Complete C++ function body or solver fragment to insert into the template.",
            "required": True,
        },
        {
            "name": "title",
            "type": "string",
            "description": "Short name for the proposed change and its expected consequences.",
            "required": True,
        },
        {
            "name": "reason",
            "type": "string",
            "description": "Short motivation for why the proposed change should help.",
            "required": True,
        },
    ]

    raw_fields = None
    for key in ("structured_output_fields", "llm_output_fields", "response_fields"):
        if config_payload.get(key) is not None:
            raw_fields = config_payload.get(key)
            break

    if isinstance(raw_fields, Mapping):
        for name, spec in raw_fields.items():
            normalized = _normalize_field_spec(name, spec)
            _upsert_field(field_specs, normalized)
    elif isinstance(raw_fields, Sequence) and not isinstance(raw_fields, (str, bytes)):
        for spec in raw_fields:
            if isinstance(spec, Mapping):
                normalized = _normalize_field_spec(spec.get("name", ""), spec)
            else:
                normalized = _normalize_field_spec(str(spec), spec)
            _upsert_field(field_specs, normalized)

    return field_specs


def _normalize_field_spec(name: str, spec: Any) -> Dict[str, Any]:
    if isinstance(spec, str):
        return {
            "name": _normalize_name(name),
            "type": spec,
            "description": "",
            "required": True,
        }
    if isinstance(spec, Mapping):
        normalized = {
            "name": _normalize_name(spec.get("name") or name),
            "type": str(spec.get("type", "string")),
            "description": str(spec.get("description", "")),
            "required": _as_bool(spec.get("required", True)),
        }
        for key in ("enum", "items", "properties", "additionalProperties", "minimum", "maximum"):
            if key in spec:
                normalized[key] = spec[key]
        return normalized
    return {
        "name": _normalize_name(name),
        "type": "string",
        "description": "",
        "required": True,
    }


def _upsert_field(field_specs: List[Dict[str, Any]], normalized: Dict[str, Any]) -> None:
    if not normalized.get("name"):
        return
    names = {field["name"] for field in field_specs}
    if normalized["name"] in names:
        for index, existing in enumerate(field_specs):
            if existing["name"] == normalized["name"]:
                field_specs[index] = normalized
                break
    else:
        field_specs.append(normalized)


def build_structured_schema(config_payload: Mapping[str, Any] | None, task_name: str | None = None) -> Dict[str, Any]:
    fields = resolve_structured_output_fields(config_payload, task_name=task_name)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for field in fields:
        field_name = str(field["name"])
        schema: Dict[str, Any] = {"type": field.get("type", "string")}
        description = str(field.get("description", "")).strip()
        if description:
            schema["description"] = description
        for key in ("enum", "items", "properties", "additionalProperties", "minimum", "maximum"):
            if key in field:
                schema[key] = field[key]
        properties[field_name] = schema
        if _as_bool(field.get("required", True)):
            required.append(field_name)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "heuristic_code_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def format_structured_output_section(config_payload: Mapping[str, Any] | None, task_name: str | None = None) -> str:
    fields = resolve_structured_output_fields(config_payload, task_name=task_name)
    required = [field["name"] for field in fields if _as_bool(field.get("required", True))]
    lines = ["Structured output requirements:"]
    lines.append("Return a valid JSON object with the fields: " + ", ".join(field["name"] for field in fields) + ".")
    if required:
        lines.append("Required fields: " + ", ".join(required) + ".")
    for field in fields:
        description = str(field.get("description", "")).strip()
        if description:
            lines.append(f"- {field['name']}: {description}")
    return "\n".join(lines)


def format_iteration_result_summary(result_rows: Sequence[Mapping[str, Any]]) -> str:
    if not result_rows:
        return ""
    lines = ["Results from the previous iteration:"]
    for row in result_rows:
        count = row.get("id", row.get("global_id", "?"))
        par2 = row.get("par2", row.get("PAR-2"))
        title = str(row.get("title", "")).strip() or "n/a"
        reason = str(row.get("reason", "")).strip() or "n/a"
        status = str(row.get("status", "accepted")).strip()
        lines.append(f"- candidate {count}: PAR-2={par2}; title={title}; reason={reason}; status={status}")
    return "\n".join(lines)


def format_baseline_summary(baseline_par2: Any, baseline_time: Any = None) -> str:
    lines = [f"Baseline iteration 0 PAR-2: {baseline_par2}"]
    if baseline_time is not None:
        lines.append(f"Baseline iteration 0 time: {baseline_time}")
    return "\n".join(lines)


# prompting.py (фрагмент с изменениями)

def build_prompt_text(
    base_prompt_text: str,
    task_name: str,
    baseline_par2: float | None = None,
    baseline_time: float | None = None,
    result_rows: Sequence[Mapping[str, Any]] | None = None,
    config_payload: Mapping[str, Any] | None = None,
    structured_output: bool = False,
    extra_sections: Sequence[str] | None = None,
) -> str:
    sections: List[str] = [base_prompt_text.strip()]
    allowed_modules = None
    if config_payload and "tasks" in config_payload:
        allowed_modules = config_payload["tasks"]
    elif config_payload and "task_name" in config_payload:
        allowed_modules = [config_payload["task_name"]]
    else:
        allowed_modules = [task_name]

    modules = list(load_heuristic_modules(config_payload or {}, allowed_modules=allowed_modules))
    module_section = format_heuristic_modules_section(modules)
    if module_section:
        sections.append(module_section)
        
    if baseline_par2 is not None:
        sections.append(format_baseline_summary(baseline_par2, baseline_time))
        
    if result_rows:
        filtered_rows = []
        
        baseline_row = next((row for row in result_rows if str(row.get("id")).lower() == "baseline"), None)
        if baseline_row:
            filtered_rows.append(baseline_row)

        if len(result_rows) > 0:
            last_row = result_rows[-1]
            if last_row != baseline_row:
                filtered_rows.append(last_row)
                
        result_section = format_iteration_result_summary(filtered_rows)
        if result_section:
            sections.append(result_section)
            
    if structured_output:
        sections.append(format_structured_output_section(config_payload or {}, task_name=task_name))
        
    for section in extra_sections or []:
        if section:
            sections.append(str(section).strip())
            
    return "\n\n".join(section for section in sections if section)

def load_prompt_text(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8")


def parse_structured_response(content: str) -> Dict[str, str]:
    try:
        data = json.loads(content)
    except Exception as exc:
        print(f"[StructuredOutput] JSON parse error: {exc}. Raw content: {content[:300]}", flush=True)
        return {"code": "", "title": "", "reason": ""}

    return {
        "code": str(data.get("code", "") or "").strip(),
        "title": str(data.get("title", "") or "").strip(),
        "reason": str(data.get("reason", "") or "").strip(),
    }
