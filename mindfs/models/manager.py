"""Local model manager, multi-format discovery (GGUF, MLX, HuggingFace), and RAM estimator."""

import gc
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mindfs.config.settings import MindFSConfig


class ModelInfo(BaseModel):
    """Metadata and RAM profile for a local model."""
    model_id: str
    model_name: str
    file_path: str
    model_format: str = "GGUF"  # GGUF, MLX, HuggingFace, Ollama
    file_size_bytes: int
    file_size_mb: float
    quantization: str = "unknown"
    parameter_scale: str = "unknown"
    context_tokens: int = 2048
    estimated_ram_mb: float
    compatibility: str  # "compatible", "warning", "exceeds_budget"
    is_active: bool = False


class ModelManager:
    """Discovers, evaluates, and safely switches local GGUF and MLX models within memory budget."""

    STANDARD_DISCOVERY_DIRS = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "lm-studio" / "models",
        Path.home() / ".cache" / "minicontainer",
        Path.home() / ".cache" / "miniai",
        Path.home() / ".ollama" / "models",
        Path.home() / ".local" / "share" / "nomic.ai" / "GPT4All",
        Path.home() / ".mlx_models",
        Path.home() / "models",
        Path.cwd() / "models",
    ]

    def __init__(self, config: MindFSConfig, llm_engine: Optional[Any] = None):
        self.config = config
        self.llm_engine = llm_engine
        self.active_model_path: Optional[str] = config.llm.model_path
        self._discovered_cache: Dict[str, ModelInfo] = {}

    @property
    def max_budget_mb(self) -> float:
        """Returns current dynamic memory budget ceiling (default 2048 MB)."""
        return self.config.resources.max_rss_mb if (self.config and self.config.resources) else 2048.0

    def parse_quantization(self, name: str) -> str:
        """Extracts quantization tag from name (e.g. Q4_K_M, Q5_K_M, Q8_0, 4-bit, 8-bit, FP16)."""
        name_upper = name.upper()
        if "4BIT" in name_upper or "4-BIT" in name_upper or "INT4" in name_upper:
            return "4-bit (MLX)" if ("MLX" in name_upper or not ".gguf" in name.lower()) else "Q4_0"
        if "8BIT" in name_upper or "8-BIT" in name_upper or "INT8" in name_upper:
            return "8-bit (MLX)" if ("MLX" in name_upper or not ".gguf" in name.lower()) else "Q8_0"
        match = re.search(r"\b(Q[2-8]_[KMS0-9_]+|F16|F32|FP16|FP32|BF16)\b", name_upper)
        if match:
            return match.group(1)
        return "Q4_K_M" if ".gguf" in name.lower() else "16-bit / FP16"

    def parse_parameter_scale(self, name: str) -> str:
        """Extracts parameter scale from name (e.g. 0.5B, 1.5B, 3B, 7B, 8B, 14B)."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*B\b", name, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1)}B"
        return "unknown"

    def estimate_ram_mb(self, file_size_bytes: int, context_tokens: int = 2048) -> float:
        """
        Estimates total process peak RSS in MB for loading this model.
        Formula: (File Size * 1.10) + (KV Cache overhead) + Base Process Overhead
        """
        file_size_mb = file_size_bytes / (1024 * 1024)
        kv_cache_mb = (context_tokens / 2048.0) * 60.0
        base_overhead_mb = 120.0  # MindFS embedding + SQLite + FAISS runtime overhead
        total_est = (file_size_mb * 1.10) + kv_cache_mb + base_overhead_mb
        return round(total_est, 1)

    def get_compatibility_rating(self, estimated_ram_mb: float) -> str:
        """
        Evaluates compatibility dynamically against current budget ceiling:
        - 'compatible'     : <= 75% of budget
        - 'warning'        : 75% - 100% of budget
        - 'exceeds_budget' : > 100% of budget
        """
        budget = self.max_budget_mb
        if estimated_ram_mb <= (0.75 * budget):
            return "compatible"
        elif estimated_ram_mb <= budget:
            return "warning"
        else:
            return "exceeds_budget"

    def scan_directories(self, additional_paths: Optional[List[str]] = None) -> List[ModelInfo]:
        """Scans filesystem locations for GGUF, MLX, and HuggingFace models."""
        dirs_to_scan: List[Path] = []

        if self.config.llm.model_path:
            p = Path(self.config.llm.model_path).expanduser().resolve()
            if p.is_dir():
                dirs_to_scan.append(p)
            elif p.parent.is_dir():
                dirs_to_scan.append(p.parent)

        for d in self.STANDARD_DISCOVERY_DIRS:
            if d.exists() and d.is_dir():
                dirs_to_scan.append(d)

        if additional_paths:
            for ap in additional_paths:
                if ap:
                    p = Path(ap).expanduser().resolve()
                    if p.exists():
                        dirs_to_scan.append(p if p.is_dir() else p.parent)

        discovered: List[ModelInfo] = []
        seen_paths = set()

        for search_dir in dirs_to_scan:
            try:
                for root, dirnames, filenames in os.walk(search_dir, followlinks=True):
                    # Skip irrelevant package and cache directories
                    if any(ign in root for ign in ("node_modules", ".git", "__pycache__", "site-packages", "venv", ".venv", "trash")):
                        continue

                    # Bounded search depth per root
                    try:
                        rel_parts = Path(root).relative_to(search_dir).parts
                        if len(rel_parts) > 5:
                            continue
                    except Exception:
                        pass

                    # 1. GGUF single-file models
                    for fn in filenames:
                        if fn.lower().endswith(".gguf"):
                            full_path = Path(root) / fn
                            canonical = str(full_path.resolve())
                            if canonical in seen_paths:
                                continue
                            seen_paths.add(canonical)

                            try:
                                sz = full_path.stat().st_size
                            except OSError:
                                continue

                            sz_mb = sz / (1024 * 1024)
                            quant = self.parse_quantization(fn)
                            params = self.parse_parameter_scale(fn)
                            ctx = self.config.llm.context_tokens or 2048
                            est_ram = self.estimate_ram_mb(sz, ctx)
                            compat = self.get_compatibility_rating(est_ram)
                            is_active = (self.active_model_path and Path(self.active_model_path).resolve() == full_path.resolve())

                            clean_name = fn.replace(".gguf", "").replace(".GGUF", "")
                            info = ModelInfo(
                                model_id=clean_name,
                                model_name=clean_name,
                                file_path=canonical,
                                model_format="GGUF",
                                file_size_bytes=sz,
                                file_size_mb=round(sz_mb, 1),
                                quantization=quant,
                                parameter_scale=params,
                                context_tokens=ctx,
                                estimated_ram_mb=est_ram,
                                compatibility=compat,
                                is_active=bool(is_active),
                            )
                            discovered.append(info)
                            self._discovered_cache[canonical] = info

                    # 2. MLX / HuggingFace Snapshot Repositories
                    if "config.json" in filenames and any(f.endswith(".safetensors") or f.endswith(".bin") or f == "weights.npz" for f in filenames):
                        full_dir = Path(root)
                        canonical = str(full_dir.resolve())
                        if canonical in seen_paths:
                            continue
                        seen_paths.add(canonical)

                        sz = sum(f.stat().st_size for f in full_dir.glob("*") if f.is_file())
                        sz_mb = sz / (1024 * 1024)

                        # Derive friendly model name
                        if "models--" in canonical:
                            repo_part = canonical.split("models--")[-1].split("/snapshots")[0].replace("--", "/")
                            clean_name = repo_part
                        else:
                            clean_name = full_dir.name

                        fmt = "MLX" if ("mlx" in canonical.lower() or "weights.npz" in filenames) else "HuggingFace / MLX"
                        quant = self.parse_quantization(canonical)
                        params = self.parse_parameter_scale(clean_name)
                        ctx = self.config.llm.context_tokens or 2048
                        est_ram = self.estimate_ram_mb(sz, ctx)
                        compat = self.get_compatibility_rating(est_ram)
                        is_active = (self.active_model_path and Path(self.active_model_path).resolve() == full_dir.resolve())

                        info = ModelInfo(
                            model_id=clean_name,
                            model_name=clean_name,
                            file_path=canonical,
                            model_format=fmt,
                            file_size_bytes=sz,
                            file_size_mb=round(sz_mb, 1),
                            quantization=quant,
                            parameter_scale=params,
                            context_tokens=ctx,
                            estimated_ram_mb=est_ram,
                            compatibility=compat,
                            is_active=bool(is_active),
                        )
                        discovered.append(info)
                        self._discovered_cache[canonical] = info

            except Exception:
                continue

        # Sort: active first, then generative LLMs before embeddings, then MLX before GGUF, then compatible, then by size
        def sort_key(m):
            is_emb = any(emb in m.model_name.lower() for emb in ("sentence-transformers", "minilm", "bge-", "e5-"))
            is_mlx = "mlx" in m.model_format.lower()
            return (not m.is_active, is_emb, not is_mlx, m.compatibility != "compatible", m.file_size_mb)

        discovered.sort(key=sort_key)
        return discovered

    def discover_models(self, custom_paths: Optional[List[str]] = None) -> List[ModelInfo]:
        """Alias for scan_directories."""
        return self.scan_directories(additional_paths=custom_paths)

    def scan_for_models(self, custom_paths: Optional[List[str]] = None) -> List[ModelInfo]:
        """Alias for scan_directories."""
        return self.scan_directories(additional_paths=custom_paths)

    def switch_model(self, model_path: str, context_tokens: int = 2048) -> Dict[str, Any]:
        """
        Safely switches active model:
        1. Unloads current LLM backend from memory
        2. Performs garbage collection to release RSS
        3. Configures and initializes new model
        """
        target = Path(model_path).expanduser().resolve()
        if not target.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # 1. Unload old model
        if self.llm_engine:
            if hasattr(self.llm_engine, "_llm") and self.llm_engine._llm is not None:
                del self.llm_engine._llm
                self.llm_engine._llm = None
            if hasattr(self.llm_engine, "_mlx_model") and self.llm_engine._mlx_model is not None:
                del self.llm_engine._mlx_model
                self.llm_engine._mlx_model = None
            if hasattr(self.llm_engine, "_mlx_tokenizer") and self.llm_engine._mlx_tokenizer is not None:
                del self.llm_engine._mlx_tokenizer
                self.llm_engine._mlx_tokenizer = None
            self.llm_engine.model_path = str(target)
            self.llm_engine.context_tokens = context_tokens

        # 2. Force GC to reclaim memory
        gc.collect()

        # 3. Update active model tracking
        self.active_model_path = str(target)
        self.config.llm.model_path = str(target)
        self.config.llm.context_tokens = context_tokens

        # 4. Attempt to initialize new model in LLMEngine
        if self.llm_engine:
            self.llm_engine._init_backend()

        if target.is_file():
            sz = target.stat().st_size
            m_name = target.stem
            fmt = "GGUF"
        else:
            sz = sum(f.stat().st_size for f in target.glob("*") if f.is_file())
            m_name = target.name
            fmt = "MLX / HuggingFace"

        est_ram = self.estimate_ram_mb(sz, context_tokens)
        compat = self.get_compatibility_rating(est_ram)

        return {
            "status": "switched",
            "model_path": str(target),
            "model_name": m_name,
            "model_format": fmt,
            "context_tokens": context_tokens,
            "file_size_mb": round(sz / (1024 * 1024), 1),
            "estimated_ram_mb": est_ram,
            "compatibility": compat,
            "is_active": True,
        }

    def get_active_model(self) -> Optional[ModelInfo]:
        """Returns metadata of currently active model."""
        if not self.active_model_path:
            return None
        p = Path(self.active_model_path).expanduser().resolve()
        if not p.exists():
            return None

        canonical = str(p)
        if canonical in self._discovered_cache:
            m = self._discovered_cache[canonical]
            m.is_active = True
            return m

        if p.is_file():
            sz = p.stat().st_size
            fmt = "GGUF"
        else:
            sz = sum(f.stat().st_size for f in p.glob("*") if f.is_file())
            fmt = "MLX / HuggingFace"

        ctx = self.config.llm.context_tokens or 2048
        est_ram = self.estimate_ram_mb(sz, ctx)
        return ModelInfo(
            model_id=p.stem if p.is_file() else p.name,
            model_name=p.stem if p.is_file() else p.name,
            file_path=canonical,
            model_format=fmt,
            file_size_bytes=sz,
            file_size_mb=round(sz / (1024 * 1024), 1),
            quantization=self.parse_quantization(p.name),
            parameter_scale=self.parse_parameter_scale(p.name),
            context_tokens=ctx,
            estimated_ram_mb=est_ram,
            compatibility=self.get_compatibility_rating(est_ram),
            is_active=True,
        )

    def get_active_model_info(self) -> Optional[ModelInfo]:
        """Alias for get_active_model."""
        return self.get_active_model()
