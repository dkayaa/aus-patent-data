"""Per-methodology instruction generation tasks."""

from __future__ import annotations

from .abstract_drafting import AbstractDraftingTask
from .base import Task
from .ipc_lcpn import IPCLCPNTask
from .ipc_reasoning import IPCReasoningTask
from .mrc import MRCTask

TASKS: dict[str, type[Task]] = {
    IPCReasoningTask.task_id: IPCReasoningTask,
    IPCLCPNTask.task_id: IPCLCPNTask,
    AbstractDraftingTask.task_id: AbstractDraftingTask,
    MRCTask.task_id: MRCTask,
}

__all__ = ["TASKS", "Task"]
