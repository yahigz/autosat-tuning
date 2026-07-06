from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str = ""
    description: str = ""
    signature: str = ""
    insertion_format: str = ""
    baseline_code: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        spec: Mapping[str, Any] | None = None,
        baseline_code: str = "",
    ) -> "TaskSpec":
        spec = dict(spec or {})
        return cls(
            name=str(name),
            label=str(spec.get("label", name.replace("_", " "))),
            description=str(spec.get("description", "")),
            signature=str(spec.get("signature", name)),
            insertion_format=str(spec.get("insertion_format", "")),
            baseline_code=str(spec.get("baseline_code", baseline_code) or baseline_code or ""),
            metadata={k: v for k, v in spec.items() if k not in {
                "label",
                "description",
                "signature",
                "insertion_format",
                "baseline_code",
            }},
        )

    def with_baseline_code(self, baseline_code: str) -> "TaskSpec":
        return TaskSpec(
            name=self.name,
            label=self.label,
            description=self.description,
            signature=self.signature,
            insertion_format=self.insertion_format,
            baseline_code=baseline_code,
            metadata=self.metadata,
        )


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().strip("/")


def task_specs_from_config(
    config_payload: Mapping[str, Any] | None,
    *,
    allowed_names: Sequence[str] | None = None,
    baseline_codes: Mapping[str, str] | None = None,
    fallback_names: Sequence[str] | None = None,
    default_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[TaskSpec]:
    config_payload = dict(config_payload or {})
    allowed = {_normalize_name(x) for x in allowed_names} if allowed_names else None
    baseline_codes = dict(baseline_codes or {})

    specs_by_name: dict[str, Mapping[str, Any]] = {
        str(name): dict(spec)
        for name, spec in (default_specs or {}).items()
    }

    raw_tasks = config_payload.get("tasks")
    if isinstance(raw_tasks, Mapping):
        for name, spec in raw_tasks.items():
            specs_by_name[_normalize_name(name)] = dict(spec) if isinstance(spec, Mapping) else {}
    elif isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)):
        for item in raw_tasks:
            if not isinstance(item, Mapping):
                continue
            name = _normalize_name(item.get("name"))
            if not name:
                continue
            specs_by_name[name] = dict(item)

    aliases = config_payload.get("heuristic_modules")
    if isinstance(aliases, Mapping):
        for name, spec in aliases.items():
            normalized = _normalize_name(name)
            base = dict(specs_by_name.get(normalized, {}))
            if isinstance(spec, Mapping):
                base.update(spec)
            specs_by_name[normalized] = base

    order: list[str] = []
    for key in ("task_order", "heuristic_module_order", "optimize_tasks"):
        raw = config_payload.get(key)
        if raw:
            items = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else [raw]
            order.extend(_normalize_name(item) for item in items)
            break
    if not order and config_payload.get("task"):
        order.append(_normalize_name(config_payload["task"]))
    if not order:
        order.extend(_normalize_name(name) for name in (fallback_names or specs_by_name.keys()))

    seen: set[str] = set()
    result: list[TaskSpec] = []
    for name in order:
        if not name or name in seen:
            continue
        if allowed is not None and name not in allowed:
            continue
        if name not in specs_by_name and name not in baseline_codes and default_specs is None:
            # Allow bare task names even when no metadata is provided.
            result.append(TaskSpec(name=name, baseline_code=baseline_codes.get(name, "")))
            seen.add(name)
            continue
        spec = specs_by_name.get(name, {})
        result.append(TaskSpec.from_mapping(name, spec, baseline_code=baseline_codes.get(name, "")))
        seen.add(name)
    return result
