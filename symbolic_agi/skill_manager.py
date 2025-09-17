# symbolic_agi/skill_manager.py
from __future__ import annotations

import json
import os
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

from symbolic_agi import config
from symbolic_agi.schemas import ActionStep, SkillModel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _step_from_any(s: Any) -> ActionStep:
    if isinstance(s, ActionStep):
        return s
    if isinstance(s, dict):
        return ActionStep(**cast(Dict[str, Any], s))  # type: ignore[misc]
    raise TypeError(f"Invalid step type: {type(s)}")


def _step_to_dict(s: Any) -> Dict[str, Any]:
    # Pydantic v2 first
    if hasattr(s, "model_dump"):
        return s.model_dump()  # type: ignore[attr-defined]
    if hasattr(s, "__dict__"):
        vars_dict = cast(Dict[str, Any], vars(s))
        return {k: v for k, v in vars_dict.items() if not k.startswith("_")}
    return cast(Dict[str, Any], s) if isinstance(s, dict) else dict(s)  # may raise; that's okay


def _skill_from_dict(d: Dict[str, Any]) -> SkillModel:
    steps_raw = cast(List[Any], d.get("action_sequence", []) or [])
    steps = [_step_from_any(x) for x in steps_raw]

    # Prefer full constructor; fall back if schema is stricter
    try:
        return SkillModel(
            id=d.get("id") or f"skill_{uuid.uuid4().hex[:16]}",
            name=d["name"],
            description=d.get("description", ""),
            action_sequence=steps,
            created_at=d.get("created_at") or _now_iso(),
            usage_count=d.get("usage_count", 0),
            effectiveness_score=d.get("effectiveness_score", 0.5),
        )
    except TypeError:
        sm = SkillModel(name=d["name"], description=d.get("description", ""), action_sequence=steps)
        for k, v in cast(Dict[str, Any], {
            "id": d.get("id") or f"skill_{uuid.uuid4().hex[:16]}",
            "created_at": d.get("created_at") or _now_iso(),
            "usage_count": d.get("usage_count", 0),
            "effectiveness_score": d.get("effectiveness_score", 0.5),
        }).items():
            if hasattr(sm, k):
                with suppress(Exception):
                    setattr(sm, k, v)
        return sm


def _skill_to_dict(s: SkillModel) -> Dict[str, Any]:
    steps = [_step_to_dict(x) for x in getattr(s, "action_sequence", [])]
    if hasattr(s, "model_dump"):
        d = s.model_dump()  # type: ignore[attr-defined]
        d["action_sequence"] = steps
        return d
    return {
        "id": getattr(s, "id", None),
        "name": getattr(s, "name", ""),
        "description": getattr(s, "description", ""),
        "action_sequence": steps,
        "created_at": getattr(s, "created_at", _now_iso()),
        "usage_count": getattr(s, "usage_count", 0),
        "effectiveness_score": getattr(s, "effectiveness_score", 0.5),
    }


def _extract_action_and_persona(st: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (action, persona) from an ActionStep or dict; else (None, None)."""
    if isinstance(st, ActionStep):
        return getattr(st, "action", None), getattr(st, "assigned_persona", None)
    if isinstance(st, dict):
        st_dict = cast(Dict[str, Any], st)
        return st_dict.get("action"), st_dict.get("assigned_persona")
    return None, None


def _format_skill_definition(skill: SkillModel, min_effectiveness: Optional[float]) -> List[str]:
    """
    Produce a list of formatted lines for a single skill, or [] if filtered out.
    Kept intentionally flat to minimize cognitive complexity.
    """
    lines: List[str] = []

    raw_eff = getattr(skill, "effectiveness_score", 0.5)
    try:
        eff = float(raw_eff)
    except Exception:
        eff = 0.5

    if min_effectiveness is not None:
        with suppress(Exception):
            if eff < float(min_effectiveness):
                return lines

    name = getattr(skill, "name", "") or ""
    desc = getattr(skill, "description", "") or ""

    # 👇 Add both lines efficiently using extend
    lines.extend([
        f'action: "{name}", description: "{desc}"',
        f"- {name}: {desc}"
    ])

    steps = getattr(skill, "action_sequence", []) or []
    for st in steps:
        action, persona = _extract_action_and_persona(st)
        if not action:
            continue
        persona_text = f" persona={persona}" if persona else ""
        lines.append(f"    * action={action}{persona_text}")

    return lines



class SkillManager:
    """
    File-backed skill store.
    - ALWAYS uses the configured path (no fallback). If the file doesn't exist, it starts empty and creates it.
    - Overrides a skill when you add one with the same name.
    - Exposes a `.skills` mapping (keyed by name) for test compatibility.
    """

    def __init__(self, skills_path: Optional[str] = None, *, file_path: Optional[str] = None) -> None:
        # Accept both `skills_path=` and legacy `file_path=` (tests use file_path)
        chosen = file_path or skills_path or getattr(config, "SKILLS_PATH", "data/skills.json")
        self.skills_path = chosen
        self._skills: List[SkillModel] = []
        self._load_skills_from_file()

    # ---------- persistence ----------

    def _reset_skills_and_ensure_dir(self) -> None:
        """Helper method to reset skills and ensure directory exists."""
        self._skills = []
        _ensure_dir(self.skills_path)
        self._save_skills_to_file()

    def _load_skills_from_file(self) -> None:
        path = self.skills_path
        if not path:
            self._skills = []
            return

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = cast(List[Dict[str, Any]], json.load(f) or [])
                self._skills = [_skill_from_dict(x) for x in data]
            except Exception:
                # Corrupt file? Start clean, overwrite on next save.
                self._reset_skills_and_ensure_dir()
        else:
            # IMPORTANT: No fallback. Configured path wins; start empty.
            self._reset_skills_and_ensure_dir()

    def _save_skills_to_file(self) -> None:
        with suppress(Exception):
            _ensure_dir(self.skills_path)
            payload = [_skill_to_dict(s) for s in self._skills]
            with open(self.skills_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---------- queries ----------

    def list_skills(self) -> List[SkillModel]:
        return list(self._skills)

    def get_skill_by_name(self, name: str) -> Optional[SkillModel]:
        needle = (name or "").strip().lower()
        return next(
            (s for s in self._skills if (getattr(s, "name", "") or "").lower() == needle),
            None
        )

    @property
    def skills(self) -> Dict[str, SkillModel]:
        """
        Mapping for compatibility with tests: name -> SkillModel.
        Tests call `sm.skills.values()`, so values must be SkillModel objects.
        """
        out: Dict[str, SkillModel] = {}
        for s in self._skills:
            key = getattr(s, "name", "") or ""
            # Latest wins on duplicate names (we override by name anyway)
            out[key] = s
        return out

    @property
    def skills_by_id(self) -> Dict[str, SkillModel]:
        """Optional: id -> SkillModel mapping (not used by tests, but handy)."""
        out: Dict[str, SkillModel] = {}
        for s in self._skills:
            if (sid := getattr(s, "id", None)):
                out[str(sid)] = s
        return out

    # ---------- mutations ----------

    def _make_skill_model(
        self,
        name: str,
        description: str,
        steps: List[ActionStep],
        *,
        id: Optional[str] = None,
        created_at: Optional[str] = None,
        usage_count: int = 0,
        effectiveness_score: float = 0.5,
    ) -> SkillModel:
        sid = id or f"skill_{uuid.uuid4().hex[:16]}"
        created = created_at or _now_iso()
        try:
            return SkillModel(
                id=sid,
                name=name,
                description=description,
                action_sequence=steps,
                created_at=created,
                usage_count=usage_count,
                effectiveness_score=effectiveness_score,
            )
        except TypeError:
            sm = SkillModel(name=name, description=description, action_sequence=steps)
            for k, v in cast(Dict[str, Any], {
                "id": sid,
                "created_at": created,
                "usage_count": usage_count,
                "effectiveness_score": effectiveness_score,
            }).items():
                if hasattr(sm, k):
                    with suppress(Exception):
                        setattr(sm, k, v)
            return sm

    def add_new_skill(self, name: str, description: str, plan: List[ActionStep]) -> SkillModel:
        """Add or override a skill by name, with minimal branching."""
        steps = [_step_from_any(p) for p in (plan or [])]
        existing = self.get_skill_by_name(name)

        # Create new
        if existing is None:
            new_skill = self._make_skill_model(name=name, description=description, steps=steps)
            self._skills.append(new_skill)
            self._save_skills_to_file()
            return new_skill

        # Try to mutate existing in-place (fast path)
        try:
            existing.description = description
            existing.action_sequence = steps
            self._save_skills_to_file()
            return existing
        except Exception:
            # Immutable model: replace it while preserving metadata
            replacement = self._make_skill_model(
                name=name,
                description=description,
                steps=steps,
                id=getattr(existing, "id", None),
                created_at=getattr(existing, "created_at", None),
                usage_count=getattr(existing, "usage_count", 0),
                effectiveness_score=getattr(existing, "effectiveness_score", 0.5),
            )
            lower_name = name.lower()
            self._skills = [
                (replacement if (getattr(s, "name", "") or "").lower() == lower_name else s)
                for s in self._skills
            ]
            self._save_skills_to_file()
            return replacement


    # ---------- optional metrics ----------

    def record_skill_use(self, name: str) -> None:
        s = self.get_skill_by_name(name)
        if not s:
            return
        try:
            s.usage_count = int(getattr(s, "usage_count", 0)) + 1
        except Exception:
            with suppress(Exception):
                setattr(s, "usage_count", int(getattr(s, "usage_count", 0)) + 1)
        self._save_skills_to_file()

    def record_skill_outcome(self, name: str, success: bool, alpha: float = 0.12) -> None:
        s = self.get_skill_by_name(name)
        if not s:
            return
        prev = float(getattr(s, "effectiveness_score", 0.5))
        target = 1.0 if success else 0.0
        new = (1 - alpha) * prev + alpha * target
        try:
            s.effectiveness_score = round(new, 3)
        except Exception:
            with suppress(Exception):
                setattr(s, "effectiveness_score", round(new, 3))
        self._save_skills_to_file()

    # ---------- planner-facing ----------

    def get_formatted_definitions(self, min_effectiveness: Optional[float] = None) -> str:
        """
        Return a plain-text block of skills and their inner actions.
        Includes a header so tests can detect the 'learned skills' section.
        """
        out_lines: List[str] = ["# LEARNED SKILLS"]
        for s in self._skills:
            out_lines.extend(_format_skill_definition(s, min_effectiveness))
        return "\n".join(out_lines)

