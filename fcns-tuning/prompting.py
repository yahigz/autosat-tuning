"""FCNS-specific prompt helpers."""
from __future__ import annotations

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
        "signature": "int Solver::UVERTEX()",
    },
    "cvertex_function": {
        "label": "Rollback vertex choice",
        "description": "Choose which colored vertices should be uncoloured when the solver needs to recover from a dead end. The number of vertices to uncolour is determined by parameter B, which is passed to the FCNS function.",
        "insertion_format": "Replace the full body of Solver::CVERTEX().",
        "signature": "int Solver::CVERTEX()",
    },
    "colour_function": {
        "label": "Color selection policy",
        "description": "Choose the best available color for the selected vertex u from the legal domain D.",
        "insertion_format": "Replace the full body of Solver::COLOUR(int u, const vector<int>& D).",
        "signature": "int Solver::COLOUR(int u, const vector<int>& D)",
    },
    "greedy_initial_solution": {
        "label": "Greedy initial solution",
        "description": "Construct an initial solution for the graph coloring problem using a greedy approach. It allows to approximate the solution quickly, which can be used as a starting point for the FCNS search. The output should contain a vector of colors assigned to each vertex in the graph (both 0-indexed). Do not be afraid to try for this function to return a totally novel solution.",
        "insertion_format": "Replace the full body of greedy_coloring(const vector<vector<int>>& graph).",
        "signature": "vector<int> greedy_coloring(const vector<vector<int>>& graph)",
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


def build_tool_schema_all(tasks, fields=None):
    return _core_build_tool_schema_all(tasks, fields=fields)


def build_structured_schema(config_payload=None, task_name=None, fields=None):
    return _core_build_structured_schema(config_payload=config_payload, task_name=task_name, fields=fields)


def build_structured_schema_all(tasks, fields=None):
    return _core_build_structured_schema_all(tasks, fields=fields)
