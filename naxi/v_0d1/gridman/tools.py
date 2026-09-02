from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from naxi.v_0d1.gridman.config import Config, RUNNING_CONFIG


CheckpointStage = Literal["pretrain", "sft", "post_train"]
CHECKPOINT_SCHEMA_VERSION = 2
STATE_PRESERVATION_CONTRACT = (
    "preserve mutable Ouro state across compatible lifecycle stages; "
    "reset only through an explicit diagnostic or recovery operation"
)
COMPATIBILITY_FIELDS = (
    "embed_dim",
    "blocks",
    "block_layers",
    "patch_size",
    "chunk_size",
)


def model_size_manifest(model: nn.Module) -> dict[str, int]:
    trainable_parameters = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    buffers = dict(model.named_buffers())
    matrix_state_elements = sum(
        buffer.numel()
        for name, buffer in buffers.items()
        if name.split(".")[-1] == "mem"
    )
    persistent_state_elements = sum(
        buffer.numel()
        for name, buffer in buffers.items()
        if _is_mutable_state_name(name)
    )
    return {
        "trainable_parameters": trainable_parameters,
        "matrix_state_elements": matrix_state_elements,
        "persistent_state_elements": persistent_state_elements,
        "repository_tracked_scale": trainable_parameters + matrix_state_elements,
    }


def print_model_parameters(model: nn.Module):
    counts = model_size_manifest(model)
    print("\n" + "=" * 60)
    print("Gridman 🤖 模型体积统计:")
    print(f" ├─ 仓库追踪规模: {counts['repository_tracked_scale'] / 1e6:.2f} M")
    print(f" ├─ 可训练参数: {counts['trainable_parameters'] / 1e6:.2f} M")
    print(f" ├─ 矩阵状态量: {counts['matrix_state_elements'] / 1e6:.2f} M")
    print(f" └─ 全部注册持久状态: {counts['persistent_state_elements'] / 1e6:.2f} M")
    print("=" * 60)


def _is_mutable_state_name(name: str) -> bool:
    leaf = name.split(".")[-1]
    return leaf in {"c_state", "c_state_queue", "mem"}


def _config_manifest(config: Config) -> dict[str, Any]:
    return {
        "name": config.name,
        "version": config.version,
        "embed_dim": config.embed_dim,
        "blocks": config.blocks,
        "block_layers": config.block_layers,
        "patch_size": config.patch_size,
        "chunk_size": config.chunk_size,
        "bptt_size": config.bptt_size,
        "tokenizer": type(config.tokenizer).__name__,
        "temporal_queue_length": 65,
    }


def _state_topology(model: nn.Module) -> dict[str, list[int]]:
    state_dict_keys = set(model.state_dict().keys())
    return {
        name: list(buffer.shape)
        for name, buffer in model.named_buffers()
        if name in state_dict_keys and _is_mutable_state_name(name)
    }


def _checkpoint_path(
    config: Config,
    stage: CheckpointStage,
    checkpoint_path: str | os.PathLike[str] | None,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    model_name = f"{config.name}_{config.version}_{stage}"
    return Path(config.checkpoint_dir) / f"{model_name}.pt"


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore checkpoint RNG state without changing model state.

    RNG restoration is deliberately separate from restoring registered Ouro
    state: the former is needed for training-trajectory equivalence, while the
    latter is handled by ``model.load_state_dict`` for every checkpoint load.
    """
    if not state:
        return
    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state.cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _environment_manifest() -> dict[str, Any]:
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def save_checkpoint(
    model: nn.Module,
    *,
    stage: CheckpointStage = "pretrain",
    config: Config = RUNNING_CONFIG,
    checkpoint_path: str | os.PathLike[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    global_step: int = 0,
    lineage_id: str | None = None,
    parent_checkpoint_id: str | None = None,
    git_sha: str | None = None,
    data_manifest_id: str | None = None,
    environment_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = _checkpoint_path(config, stage, checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = str(uuid.uuid4())
    lineage_id = lineage_id or str(uuid.uuid4())

    checkpoint: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "lineage_id": lineage_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "stage": stage,
        "global_step": int(global_step),
        "git_sha": git_sha,
        "model_config": _config_manifest(config),
        "state_topology": _state_topology(model),
        "model_size": model_size_manifest(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _rng_state(),
        "data_manifest_id": data_manifest_id,
        "environment_manifest": environment_manifest or _environment_manifest(),
        "state_preservation_contract": STATE_PRESERVATION_CONTRACT,
    }
    torch.save(checkpoint, path)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "lineage_id": lineage_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "stage": stage,
        "global_step": int(global_step),
        "checkpoint_path": str(path),
    }


def _validate_compatibility(
    checkpoint: dict[str, Any],
    model: nn.Module,
    config: Config,
) -> None:
    stored_config = checkpoint.get("model_config")
    if stored_config:
        current_config = _config_manifest(config)
        mismatches = {
            field: (stored_config.get(field), current_config.get(field))
            for field in COMPATIBILITY_FIELDS
            if stored_config.get(field) != current_config.get(field)
        }
        if mismatches:
            details = ", ".join(
                f"{field}: checkpoint={old!r}, current={new!r}"
                for field, (old, new) in mismatches.items()
            )
            raise ValueError(
                "Checkpoint belongs to an incompatible model lineage; "
                f"implicit state migration is disabled ({details})."
            )

    stored_topology = checkpoint.get("state_topology")
    if stored_topology and stored_topology != _state_topology(model):
        raise ValueError(
            "Checkpoint state topology does not match the current model; "
            "implicit state migration is disabled."
        )


def load_checkpoint(
    model: nn.Module,
    *,
    stage: CheckpointStage = "pretrain",
    config: Config = RUNNING_CONFIG,
    checkpoint_path: str | os.PathLike[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_training_state: bool = False,
    restore_rng: bool = False,
    need_print: bool = True,
) -> dict[str, Any]:
    path = _checkpoint_path(config, stage, checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"⚠️ 未找到检查点: {path}")

    checkpoint = torch.load(path, map_location=config.device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")

    is_legacy = "schema_version" not in checkpoint
    if not is_legacy:
        _validate_compatibility(checkpoint, model, config)
    model.load_state_dict(checkpoint["model_state_dict"])

    if restore_training_state:
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith("_state_dict") and key != "rng_state"
    }
    metadata.update(
        {
            "checkpoint_path": str(path),
            "legacy_manifest": is_legacy,
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
            "rng_state": checkpoint.get("rng_state"),
        }
    )

    if need_print:
        suffix = " (legacy manifest incomplete)" if is_legacy else ""
        print(f"🔄 已从 {path} 加载模型参数与注册状态{suffix}")
    return metadata
