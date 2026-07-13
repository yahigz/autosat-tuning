"""FCNS-specific prompt helpers."""
from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from autosat_core.prompting import *  # noqa: F401,F403
from autosat_core.prompting import (  # noqa: F401
    build_prompt_text as _core_build_prompt_text,
    build_prompt_text_for_tasks as _core_build_prompt_text_for_tasks,
    build_structured_schema as _core_build_structured_schema,
    build_structured_schema_all as _core_build_structured_schema_all,
    build_tool_schema as _core_build_tool_schema,
    build_tool_schema_all as _core_build_tool_schema_all,
    load_heuristic_modules as _core_load_heuristic_modules,
)


DEFAULT_HEURISTIC_MODULES = {
    "uvertex_function": {
        "label": "Uncoloured vertex choice",
        "description": "Select the next uncoloured vertex to colour during the search process.",
        "insertion_format": "Replace the full body of Solver::UVERTEX().",
        "signature": "int Solver::UVERTEX() const",
    },
    "cvertex_function": {
        "label": "Rollback vertex choice",
        "description": "Choose which colored vertices should be uncoloured when the solver needs to recover from a dead end. The number of vertices to uncolour is determined by parameter B, which is passed to the FCNS function.",
        "insertion_format": "Replace the full body of Solver::CVERTEX().",
        "signature": "int Solver::CVERTEX() const",
    },
    "colour_function": {
        "label": "Color selection policy",
        "description": "Choose the best available color for the selected vertex u from the legal domain D.",
        "insertion_format": "Replace the full body of Solver::COLOUR(int u, const vector<int>& D).",
        "signature": "int Solver::COLOUR(int u, const vector<int>& D) const",
    },
    "greedy_initial_solution": {
        "label": "Greedy initial solution",
        "description": "Construct an initial solution for the graph coloring problem using a greedy approach. It allows to approximate the solution quickly, which can be used as a starting point for the FCNS search. The output should contain a vector of colors assigned to each vertex in the graph (both 0-indexed). Do not be afraid to try a totally novel solution here, it is important to check completely different approaches.",
        "insertion_format": "Replace the full body of Solver::greedy_coloring(const vector<vector<int>>& graph).",
        "signature": "vector<int> Solver::greedy_coloring(const vector<vector<int>>& graph)",
    },
}


def load_heuristic_modules(config_payload, allowed_modules=None):
    return _core_load_heuristic_modules(
        config_payload,
        allowed_modules=allowed_modules,
        default_modules=DEFAULT_HEURISTIC_MODULES,
    )


def build_prompt_text_for_tasks(*args, **kwargs):
    kwargs.setdefault("default_modules", DEFAULT_HEURISTIC_MODULES)
    return _core_build_prompt_text_for_tasks(*args, **kwargs)


def build_prompt_text(*args, **kwargs):
    kwargs.setdefault("default_modules", DEFAULT_HEURISTIC_MODULES)
    return _core_build_prompt_text(*args, **kwargs)


def build_tool_schema(config_payload=None, task_name=None, fields=None):
    return _core_build_tool_schema(config_payload=config_payload, task_name=task_name, fields=fields)


def build_tool_schema_all(tasks, fields=None, enable_exploration: bool = False):
    schema = _core_build_tool_schema_all(tasks, fields=fields)
    if enable_exploration:
        schema["input_schema"]["properties"]["exploration_code"] = {
            "type": "string",
            "description": (
                "CRITICAL: Contains ONLY valid, runnable Python code. "
                "Do NOT add markdown fences, DO NOT add C++ comments (like //), "
                "DO NOT add explanations or prose. If no new statistics are needed, "
                "return a completely empty string.\n"
                "Function signature must be exactly:\n"
                "def get_statistics(n: int, m: int, adj_list: list[list[int]]) -> dict[str, float]:"
            ),
        }
        schema["input_schema"]["required"] = list(schema["input_schema"].get("required", [])) + ["exploration_code"]
    return schema


def build_structured_schema(config_payload=None, task_name=None, fields=None):
    return _core_build_structured_schema(config_payload=config_payload, task_name=task_name, fields=fields)


def build_structured_schema_all(tasks, fields=None, enable_exploration: bool = False):
    schema = _core_build_structured_schema_all(tasks, fields=fields)
    if enable_exploration:
        inner_schema = schema["json_schema"]["schema"]
        inner_schema["properties"]["exploration_code"] = {"type": "string"}
        inner_schema["required"] = list(inner_schema.get("required", [])) + ["exploration_code"]
    return schema


def parse_multi_response(content: str, tasks: Sequence[str]) -> Dict[str, Any]:
    """Extended parse that also extracts exploration_code into '__exploration__' key."""
    empty: Dict[str, Any] = {t: {"code": "", "title": "", "reason": ""} for t in tasks}
    empty["__exploration__"] = {"code": ""}
    try:
        data = json.loads(content)
    except Exception as exc:
        print(f"[StructuredOutput][all] JSON parse error: {exc}. Raw: {content[:300]}", flush=True)
        return empty

    impls = data.get("implementations", [])
    if not isinstance(impls, list):
        print("[StructuredOutput][all] 'implementations' is not a list.", flush=True)
        return empty

    result: Dict[str, Any] = {}
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

    result["__exploration__"] = {"code": str(data.get("exploration_code", "") or "").strip()}
    return result
