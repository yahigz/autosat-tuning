"""Generic marker-based solver source helpers."""
from __future__ import annotations

import re
from pathlib import Path


_MARKER_RE = re.compile(r"<--([A-Za-z_]\w*)-->")


def discover_available_tasks(baseline_cpp: Path) -> list[str]:
    if not baseline_cpp.exists():
        raise FileNotFoundError(f"Baseline solver not found: {baseline_cpp}")
    text = baseline_cpp.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    tasks: list[str] = []
    for match in _MARKER_RE.finditer(text):
        name = match.group(1)
        seen[name] = seen.get(name, 0) + 1
    for name, count in seen.items():
        if count >= 2:
            tasks.append(name)
    return tasks


def extract_baseline_section(baseline_text: str, marker_name: str) -> str:
    pattern = re.compile(
        r"<--" + re.escape(marker_name) + r"-->\n(.*?)\n<--" + re.escape(marker_name) + r"-->",
        re.DOTALL,
    )
    match = pattern.search(baseline_text)
    return match.group(1).strip() if match else ""


def build_solver_source(
    codes: dict[str, str],
    baseline_cpp: Path,
    substitutions: dict[str, str] | None = None,
) -> str:
    text = baseline_cpp.read_text(encoding="utf-8")

    def _replace_section(marker_name: str, replacement: str) -> None:
        nonlocal text
        pattern = re.compile(
            r"<--" + re.escape(marker_name) + r"-->\n(.*?)\n<--" + re.escape(marker_name) + r"-->",
            re.DOTALL,
        )
        if not pattern.search(text):
            return
        text = pattern.sub(replacement.strip(), text, count=1)

    for name in discover_available_tasks(baseline_cpp):
        provided = codes.get(name, "")
        if provided and len(provided.strip()) >= 10:
            _replace_section(name, provided.strip())
        else:
            baseline_code = extract_baseline_section(text, name)
            _replace_section(name, baseline_code)

    for key, value in (substitutions or {}).items():
        text = re.sub(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", str(value), text)
    return text
