"""Resource management, memory budgeting, and diagnostic tracking for MindFS."""

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import os
import platform
import resource
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import psutil

from mindfs.config.settings import MindFSConfig, ResourceConfig


class OperationDiagnostic(BaseModel):
    operation: str
    peak_rss_mb: float
    current_rss_mb: float
    duration_seconds: float
    files_processed: int = 0
    bytes_processed: int = 0
    chunks_processed: int = 0
    errors: int = 0
    status: str = "COMPLETED"
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ResourceManager:
    """Monitors process memory (RSS) and tracks operation diagnostics."""

    def __init__(self, config: Optional[MindFSConfig] = None):
        self.config = config or MindFSConfig()
        self.max_rss_mb = self.config.resources.max_rss_mb
        self.history: List[OperationDiagnostic] = []
        self._loaded_models: Dict[str, Any] = {}
        self._active_processor: Optional[str] = None

    @staticmethod
    def get_current_rss_mb() -> float:
        """Returns the current Resident Set Size (RSS) in megabytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def get_peak_rss_mb() -> float:
        """Returns the process lifetime peak RSS in megabytes."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes, on Linux in kilobytes
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024

    def available_budget_mb(self) -> float:
        """Returns remaining estimated memory budget in MB."""
        current = self.get_current_rss_mb()
        return max(0.0, self.max_rss_mb - current)

    def set_budget(self, max_rss_mb: float) -> float:
        """Dynamically updates the memory budget ceiling in MB."""
        self.max_rss_mb = float(max_rss_mb)
        if self.config and hasattr(self.config, "resources"):
            self.config.resources.max_rss_mb = float(max_rss_mb)
        return self.max_rss_mb

    def is_safe_to_allocate(self, estimated_mb: float) -> bool:
        """Checks whether allocating estimated_mb would breach the budget."""
        current = self.get_current_rss_mb()
        return (current + estimated_mb) <= self.max_rss_mb

    def register_model(self, name: str, model_instance: Any) -> None:
        """Tracks a resident ML model in memory."""
        self._loaded_models[name] = model_instance

    def unload_model(self, name: str) -> None:
        """Unloads a registered model and triggers garbage collection."""
        if name in self._loaded_models:
            del self._loaded_models[name]
            gc.collect()

    def unload_all_optional_models(self) -> None:
        """Unloads all optional models."""
        keys = list(self._loaded_models.keys())
        for k in keys:
            del self._loaded_models[k]
        gc.collect()

    @contextmanager
    def track_operation(self, operation: str):
        """Context manager to measure and log duration, RSS, and stats for an operation."""
        start_time = time.perf_counter()
        initial_peak = self.get_peak_rss_mb()
        diag = OperationDiagnostic(
            operation=operation,
            peak_rss_mb=initial_peak,
            current_rss_mb=self.get_current_rss_mb(),
            duration_seconds=0.0
        )
        try:
            yield diag
            diag.status = "COMPLETED"
        except Exception as exc:
            diag.status = "FAILED"
            diag.errors += 1
            diag.details["error"] = str(exc)
            raise
        finally:
            end_time = time.perf_counter()
            diag.duration_seconds = round(end_time - start_time, 4)
            diag.peak_rss_mb = round(self.get_peak_rss_mb(), 2)
            diag.current_rss_mb = round(self.get_current_rss_mb(), 2)
            self.history.append(diag)

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive resource manager diagnostics."""
        return {
            "current_rss_mb": round(self.get_current_rss_mb(), 2),
            "peak_rss_mb": round(self.get_peak_rss_mb(), 2),
            "budget_max_rss_mb": self.max_rss_mb,
            "budget_remaining_mb": round(self.available_budget_mb(), 2),
            "loaded_models": list(self._loaded_models.keys()),
            "active_processor": self._active_processor,
            "operations_recorded": len(self.history),
            "history": [d.model_dump() for d in self.history]
        }

