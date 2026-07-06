from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tasks import TaskSpec


@dataclass
class MarkerSolverAdapter:
    name: str
    baseline_cpp: Path
    template_cpp: Path | None = None
    task_specs: Sequence[TaskSpec] = field(default_factory=tuple)
    open_marker: str = "<--{name}-->"
    close_marker: str = "<--{name}-->"

    def _marker_pattern(self, task_name: str) -> str:
        marker = re.escape(f"<--{task_name}-->")
        return rf"^(?:\s*(?://|/\*)\s*)?{marker}(?:\s*\*/)?\s*$"

    def _section_pattern(self, task_name: str) -> re.Pattern[str]:
        marker_line = self._marker_pattern(task_name)
        return re.compile(
            rf"{marker_line}\n(.*?)(?:\n{marker_line})",
            re.DOTALL | re.MULTILINE,
        )

    def available_task_names(self) -> list[str]:
        if not self.baseline_cpp.exists():
            raise FileNotFoundError(f"Baseline solver not found: {self.baseline_cpp}")
        text = self.baseline_cpp.read_text(encoding="utf-8")
        marker_re = re.compile(r"(?:^|\n)\s*(?://|/\*)?\s*<--([A-Za-z_]\w*)-->\s*(?:\*/)?\s*(?=$|\n)")
        seen: dict[str, int] = {}
        tasks: list[str] = []
        for match in marker_re.finditer(text):
            task_name = match.group(1)
            seen[task_name] = seen.get(task_name, 0) + 1
        for task_name, count in seen.items():
            if count >= 2:
                tasks.append(task_name)
        return tasks

    def task_names(self) -> list[str]:
        if self.task_specs:
            return [task.name for task in self.task_specs]
        return self.available_task_names()

    def task_map(self) -> dict[str, TaskSpec]:
        tasks = self.task_specs or [TaskSpec(name=name) for name in self.available_task_names()]
        return {task.name: task for task in tasks}

    def baseline_text(self) -> str:
        return self.baseline_cpp.read_text(encoding="utf-8")

    def extract_baseline_section(self, task_name: str, baseline_text: str | None = None) -> str:
        text = baseline_text if baseline_text is not None else self.baseline_text()
        pattern = self._section_pattern(task_name)
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    def render_source(
        self,
        updates: Mapping[str, str],
        substitutions: Mapping[str, Any] | None = None,
        baseline_text: str | None = None,
    ) -> str:
        text = baseline_text if baseline_text is not None else self.baseline_text()

        def _replace_section(task_name: str, replacement: str) -> None:
            nonlocal text
            pattern = self._section_pattern(task_name)
            if not pattern.search(text):
                return
            text = pattern.sub(replacement.strip(), text, count=1)

        for task_name in self.task_names():
            provided = str(updates.get(task_name, "") or "").strip()
            if len(provided) >= 10:
                _replace_section(task_name, provided)
            else:
                baseline_code = self.extract_baseline_section(task_name, baseline_text=text)
                _replace_section(task_name, baseline_code)

        for key, value in (substitutions or {}).items():
            text = re.sub(r"\{\{\s*" + re.escape(str(key)) + r"\s*\}\}", str(value), text)
        return text

    def infer_task_name(self, code: str) -> str | None:
        text = str(code or "").strip()
        match = re.search(r"void\s+Solver::([A-Za-z_]\w*)\s*\(", text)
        if match:
            function_name = match.group(1)
            return {
                "restart": "restart_function",
                "rephase": "rephase_function",
                "bump_var": "bump_var_function",
            }.get(function_name)
        if "restart();" in text:
            return "restart_condition"
        if "rephase();" in text:
            return "rephase_condition"
        return None

    def write_template(self, target_path: Path, updates: Mapping[str, str], substitutions: Mapping[str, Any] | None = None) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(self.render_source(updates, substitutions=substitutions), encoding="utf-8")
