"""Reusable core for LLM-driven solver optimization."""

from .common import (
    clean_files,
    collect_results,
    collect_results_eval,
    copy_folder,
    delete_InfiniteLoopInst,
    find_key_for_value,
    get_batch_id,
    get_code,
    get_results_root,
    get_temp_root,
    revise_file,
)
from .marker_adapter import MarkerSolverAdapter
from .prompting import build_prompt_text_for_tasks
from .tasks import TaskSpec, task_specs_from_config

__all__ = [
    "clean_files",
    "collect_results",
    "collect_results_eval",
    "copy_folder",
    "delete_InfiniteLoopInst",
    "find_key_for_value",
    "get_batch_id",
    "get_code",
    "get_results_root",
    "get_temp_root",
    "MarkerSolverAdapter",
    "build_prompt_text_for_tasks",
    "TaskSpec",
    "task_specs_from_config",
    "revise_file",
]
