import os
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist
from naxi.v_0d2.gridman.config import Config, RUNNING_CONFIG


RANK_LOCAL_BUFFERS = frozenset({'c_state', 'c_state_queue'})


def print_model_parameters(model: nn.Module):
    trainable_params = 0
    mem_buffers_size = 0
    
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            
    for name, buffer in model.named_buffers():
        if name.split('.')[-1] == 'mem':
            mem_buffers_size += buffer.numel()
    
    total_tracked = trainable_params + mem_buffers_size
    mem_buffers_size += 64 * model.embed_dim
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"Gridman 🤖 模型体积统计:")
    print(f" ├─ 总规模: {total_tracked / 1e6:.2f} M")
    print(f" ├─ 可训练参数: {trainable_params / 1e6:.2f} M")
    print(f" └─ 记忆状态量: {mem_buffers_size / 1e6:.2f} M")
    print("="*60)


def save_checkpoint(
        model: nn.Module,
        is_sft: bool = False,
        config: Config = RUNNING_CONFIG,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        step: int = 0,
        dataloader: Any = None,
):
    """保存完整训练状态: 权重 + 优化器 + 调度器 + 步数 + 各 rank 的 RNG/数据流/私有 buffer。

    buffer 分两类处理:
    - 全局同步 buffer(如 mem, mem_sync 时跨 rank 平均): 各 rank 一致, 只存 rank0 的值
    - rank 私有 buffer(c_state / c_state_queue): 按 rank 收集进 runtime_states, 各自恢复

    所有 rank 都必须调用(内部要收集各 rank 的运行时状态), 仅 rank0 写盘。
    权重在保存时剥离 torch.compile 的 _orig_mod. 前缀; 临时文件 + os.replace 原子写入。
    """
    use_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_dist else 0
    stage = 'sft' if is_sft else 'pretrain'

    checkpoint = {
        'model_state_dict': {k.removeprefix('_orig_mod.'): v for k, v in model.state_dict().items()},
        'step': step,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 收集本 rank 私有的运行时 buffer(c_state / c_state_queue 等)
    rank_local_buffers = {}
    for k, v in model.named_buffers():
        kb = k.removeprefix('_orig_mod.')
        if kb.split('.')[-1] in RANK_LOCAL_BUFFERS:
            rank_local_buffers[kb] = v.detach().cpu().clone()

    # 汇总各 rank 的运行时状态(RNG + 数据流读取位置 + 私有 buffer), 恢复时按 rank 各自取回
    runtime_state = {
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'dataloader': dataloader.state_dict() if dataloader is not None else None,
        'buffers': rank_local_buffers,
    }
    if use_dist:
        runtime_states = [None] * dist.get_world_size()
        dist.all_gather_object(runtime_states, runtime_state)
    else:
        runtime_states = [runtime_state]
    checkpoint['runtime_states'] = runtime_states

    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(config.checkpoint_dir, f'{config.name}_{config.version}_{stage}.pt')
        tmp_path = checkpoint_path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)


def load_checkpoint(
        model: nn.Module,
        is_sft: bool = False,
        need_print: bool = True,
        config: Config = RUNNING_CONFIG,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        dataloader: Any = None,
) -> int:
    """断点续训统一入口, 返回已完成的 step 数(0 表示从头训练)。

    - 本阶段检查点存在: 完整恢复 模型/优化器/调度器/本 rank 的 RNG、数据流与私有 buffer
    - is_sft=True 且无 SFT 检查点: 自动回退加载 pretrain 权重(仅模型), 返回 0
    - pretrain 无检查点: 返回 0, 从头训练, 不再抛异常

    所有 rank 都需调用(各自读取同一存档, 按 rank 取回自己的运行时状态)。
    model 请传入未编译的原始模块(compile/DDP 只是包装, 共享同一批参数)。
    """
    use_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_dist else 0
    stage = 'sft' if is_sft else 'pretrain'
    checkpoint_path = os.path.join(config.checkpoint_dir, f'{config.name}_{config.version}_{stage}.pt')

    resume = os.path.exists(checkpoint_path)

    if not resume:
        if is_sft:
            # SFT 首次启动: 回退到 pretrain 权重做初始化
            pretrain_path = os.path.join(config.checkpoint_dir, f'{config.name}_{config.version}_pretrain.pt')
            if not os.path.exists(pretrain_path):
                raise FileNotFoundError(f'⚠️ 未找到检查点: {checkpoint_path}, 也未找到预训练权重: {pretrain_path}')
            checkpoint_path = pretrain_path
        else:
            if need_print:
                print(f'⚠️ 未找到检查点, 从头开始训练: {checkpoint_path}')
            return 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # 兼容 key 带 _orig_mod. 前缀的旧存档
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model_state_dict'].items()})

    if not resume:
        # SFT 权重初始化: 只取模型权重, 不恢复任何训练状态
        if need_print:
            print(f'✅ 已从 {checkpoint_path} 加载预训练权重, SFT 将从头开始训练')
        return 0

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        # load_state_dict 会自动把 state 迁移到参数所在设备
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 恢复本 rank 的 RNG、数据流读取位置与私有 buffer
    runtime_states = checkpoint.get('runtime_states')
    if runtime_states:
        rs = runtime_states[rank] if rank < len(runtime_states) else runtime_states[0]
        if rs.get('rng_state') is not None:
            torch.set_rng_state(rs['rng_state'])
        if rs.get('cuda_rng_state_all') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rs['cuda_rng_state_all'])
        if dataloader is not None and rs.get('dataloader') is not None:
            dataloader.load_state_dict(rs['dataloader'])
        # 用本 rank 保存的运行时 buffer 覆盖回模型(load_state_dict 载入的是 rank0 的值)
        if rs.get('buffers'):
            named_buffers = {k.removeprefix('_orig_mod.'): v for k, v in model.named_buffers()}
            for k, v in rs['buffers'].items():
                buf = named_buffers.get(k)
                if buf is not None and buf.shape == v.shape:
                    buf.copy_(v)

    start_step = checkpoint.get('step', 0)
    if need_print:
        print(f'🔄 已从 {checkpoint_path} 加载, 当前 step: {start_step}')
    return start_step