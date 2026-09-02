import os
import torch.distributed as dist
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from naxi.v_0d2.gridman.config import RUNNING_CONFIG
from naxi.v_0d2.gridman.core import Gridman
from naxi.v_0d2.gridman.dataloader import StreamLoader
from naxi.v_0d2.gridman.tools import save_checkpoint, load_checkpoint, print_model_parameters
from torch.utils.tensorboard import SummaryWriter


def reduce_value(value: torch.Tensor):
    if not dist.is_initialized():
        return value
    val = value.data.clone()
    # 求和
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    # 真实平均值
    return val / dist.get_world_size()


def train_model(is_sft: bool = False, grad_accum_steps: int = 1):
    config = RUNNING_CONFIG
    dtype = config.dtype
    # 初始化多卡环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    chunk_size = config.chunk_size
    bptt_size = config.bptt_size

    if is_sft:
        mode_name = 'SFT'
        model_name = f'{config.name}_sft'
        dataset_file = config.sft_train_file
        lr = config.sft_lr
        steps = config.sft_steps
    else:
        mode_name = 'PRE-TRAIN'
        model_name = f'{config.name}_pretrain'
        dataset_file = config.pretrain_train_file
        lr = config.pretrain_lr
        steps = config.pretrain_steps

    dataloader = StreamLoader(
        patch_size=config.patch_size, 
        chunk_size=chunk_size, 
        datasets=dataset_file,
        is_sft=is_sft,
        rank=local_rank,
        world_size=world_size
    )

    grid_man = Gridman(config).to(device)
    raw_grid_man = grid_man  # 保留未编译的原始模块引用, 供加载权重使用
    writer = None

    if local_rank == 0:
        print(f'🚀 正在初始化 {mode_name}...')
        print_model_parameters(grid_man)
        print('\n' + '>'*25 + f' 开始极速流式 {mode_name} ' + '<'*25)

        log_dir = os.path.join('log', model_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f'📊 TensorBoard 日志将保存至: {log_dir}')

    grid_man = torch.compile(grid_man)
    # 此处为强制类型标记
    grid_man: Gridman = DDP(grid_man, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=(bptt_size==1))
    grid_man_module: Gridman = grid_man.module

    optimizer = torch.optim.AdamW(grid_man.parameters(), lr=lr)

    total_update_steps = steps // (bptt_size * grad_accum_steps)

    num_warmup_steps = int(total_update_steps * 0.05) 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_update_steps
    )

    # ================= 断点重训 =================
    # 本阶段检查点存在 -> 完整恢复 模型/优化器/调度器/RNG/数据流, 返回已训练 step
    # is_sft=True 且无 SFT 检查点 -> 自动回退加载 pretrain 权重, 返回 0
    # 无检查点 -> 返回 0, 从头训练
    start_step = load_checkpoint(
        raw_grid_man, is_sft,
        need_print=(local_rank == 0),
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
    )
    dist.barrier()  # 确保各 rank 状态一致后再开训

    loss_acc = torch.tensor(0.0, device=device)
    loss_acc_log = torch.tensor(0.0, device=device)

    optimizer.zero_grad()

    save_every = 3600

    for step in range(start_step, steps): 
        grid_man.train()
        step_true = step + 1
        
        # 接收 Token 和 Mask
        input_patches, mask_patches = dataloader.get_batch()
        input_patches = input_patches.to(device)
        mask_patches = mask_patches.to(device)
        
        # 构造 Input 和 Target
        inputs = input_patches[:, :-1]   
        targets = input_patches[:, 1:].clone()  # 防止修改原 tensor
        target_masks = mask_patches[:, 1:]      # 与 targets 对应
        
        targets[target_masks == 0] = -100

        # 最后一次 forward 触发同步
        is_bptt_step = (step_true % bptt_size == 0)
        is_update_step = (step_true % (bptt_size * grad_accum_steps) == 0)

        sync_context = grid_man.no_sync() if not is_update_step  else nullcontext()
        
        with sync_context:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = grid_man(inputs)
                grid_man_module.core_ouro.mem_sync()

                if (targets != -100).any():
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        targets.reshape(-1),
                        ignore_index=-100 
                    )
                else: 
                    loss = logits.sum() * 0.0
            
                loss_acc = loss_acc + loss

            with torch.no_grad():
                dist_loss = reduce_value(loss)
                loss_acc_log = loss_acc_log + dist_loss

            if is_bptt_step:
                loss_for_backward = loss_acc / (bptt_size * grad_accum_steps)
                loss_for_backward.backward()

                grid_man_module.core_ouro.mem_detach()
                loss_acc = torch.tensor(0.0, device=device)

        if is_update_step:
            total_norm = torch.nn.utils.clip_grad_norm_(grid_man.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if local_rank == 0 and step_true % (bptt_size * grad_accum_steps * 10) == 0:
                avg_loss = loss_acc_log.item() / (bptt_size * grad_accum_steps)
                writer.add_scalar('Train/Loss', avg_loss, step)
                writer.add_scalar('Train/Grad_Norm', total_norm, step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
                print(f'\n📌 Step {step_true} | Total Loss = {avg_loss:.4f}')

            loss_acc_log = torch.tensor(0.0, device=device)

        if step_true % save_every == 0:
            # 所有 rank 都必须进入(内部要收集各 rank 的 RNG/数据流状态), 仅 rank0 写盘
            save_checkpoint(
                grid_man_module, is_sft,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step_true,
                dataloader=dataloader,
            )
            dist.barrier()  # 等 rank0 写完, 避免其他 rank 超前进入下一个集合通信

    # ================= 训练结束: 保存最终检查点 =================
    save_checkpoint(
        grid_man_module, is_sft,
        optimizer=optimizer,
        scheduler=scheduler,
        step=steps,
        dataloader=dataloader,
    )
    dist.barrier()

    if local_rank == 0:
        writer.close()
        print(f'✅ {mode_name} 训练完成, 最终检查点已保存')

    dist.destroy_process_group()