"""Per-methodology instruction generation tasks."""

from __future__ import annotations

from .abstract_drafting import AbstractDraftingTask
from .base import Task
from .legal_reasoning import LegalReasoningTask
from .mrc import MRCTask
from .patent_drafting import PatentDraftingTask

TASKS: dict[str, type[Task]] = {
    LegalReasoningTask.task_id: LegalReasoningTask,
    AbstractDraftingTask.task_id: AbstractDraftingTask,
    PatentDraftingTask.task_id: PatentDraftingTask,
    MRCTask.task_id: MRCTask,
}

__all__ = ["TASKS", "Task"]
