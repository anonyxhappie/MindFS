"""Tests for Local Model Manager, GGUF and MLX discovery, dynamic RAM budget, and model switching."""

import json
from pathlib import Path
import pytest

from mindfs.agent.llm import LLMEngine
from mindfs.models.manager import ModelInfo, ModelManager


def test_model_manager_ram_estimation_and_dynamic_budget(mindfs_env):
    cfg = mindfs_env["config"]
    manager = ModelManager(config=cfg)

    # 1. Default budget is 2048.0 MB (2.0 GB)
    assert manager.max_budget_mb == 2048.0

    # 2. Test quantization and parameter parsing
    assert manager.parse_quantization("llama-3-8b-instruct.Q4_K_M.gguf") == "Q4_K_M"
    assert manager.parse_quantization("qwen2.5-1.5b-q8_0.gguf") == "Q8_0"
    assert manager.parse_quantization("mlx-community-4bit") == "4-bit (MLX)"
    assert manager.parse_parameter_scale("llama-3-8b-instruct.Q4_K_M.gguf") == "8B"
    assert manager.parse_parameter_scale("qwen2.5-1.5b.gguf") == "1.5B"

    # 3. Test RAM estimation under 2.0 GB default budget
    # 500 MB model file (~0.5B model)
    est_500mb = manager.estimate_ram_mb(500 * 1024 * 1024, context_tokens=2048)
    assert est_500mb < 1200.0
    assert manager.get_compatibility_rating(est_500mb) == "compatible"

    # 1500 MB model file (~1.5B/3B Q4) -> under 2.0 GB is warning or compatible
    est_1500mb = manager.estimate_ram_mb(1500 * 1024 * 1024, context_tokens=2048)
    assert est_1500mb < 2048.0
    assert manager.get_compatibility_rating(est_1500mb) in ("compatible", "warning")

    # 5000 MB model file -> exceeds 2.0 GB budget
    est_5gb = manager.estimate_ram_mb(5000 * 1024 * 1024, context_tokens=4096)
    assert est_5gb > 2048.0
    assert manager.get_compatibility_rating(est_5gb) == "exceeds_budget"

    # 4. Dynamically expand budget to 8.0 GB (8192 MB)
    cfg.resources.max_rss_mb = 8192.0
    assert manager.max_budget_mb == 8192.0
    assert manager.get_compatibility_rating(est_5gb) == "compatible"


def test_model_manager_mlx_and_gguf_discovery(mindfs_env, tmp_path):
    cfg = mindfs_env["config"]
    llm_engine = LLMEngine(cfg)
    manager = ModelManager(config=cfg, llm_engine=llm_engine)

    # 1. Create dummy GGUF model file
    models_dir = tmp_path / "custom_models"
    models_dir.mkdir()
    
    dummy_gguf = models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    dummy_gguf.write_bytes(b"GGUF_DUMMY_HEADER" + b"\x00" * 1024 * 1024)  # 1 MB

    # 2. Create dummy MLX / HuggingFace model repo folder
    mlx_model_dir = tmp_path / "mlx_models" / "mlx-community" / "Qwen2.5-0.5B-Instruct-4bit"
    mlx_model_dir.mkdir(parents=True)
    (mlx_model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 896}))
    (mlx_model_dir / "model.safetensors").write_bytes(b"\x00" * 1024 * 1024)

    # Scan both custom paths
    discovered = manager.scan_directories([str(models_dir), str(tmp_path / "mlx_models")])
    assert len(discovered) >= 2
    
    found_names = [m.model_name for m in discovered]
    assert "qwen2.5-0.5b-instruct-q4_k_m" in found_names
    assert "Qwen2.5-0.5B-Instruct-4bit" in found_names

    mlx_entry = next(m for m in discovered if "Qwen2.5-0.5B-Instruct-4bit" in m.model_name)
    assert mlx_entry.model_format in ("MLX", "HuggingFace / MLX")

    # Test model switching to MLX model
    switch_res = manager.switch_model(str(mlx_model_dir), context_tokens=2048)
    assert switch_res["status"] == "switched"
    assert manager.active_model_path == str(mlx_model_dir.resolve())
