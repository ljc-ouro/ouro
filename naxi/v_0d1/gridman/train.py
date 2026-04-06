import os
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn.functional as F

from naxi.v_0d1.gridman.config import RUNNING_CONFIG
from naxi.v_0d1.gridman.core import Gridman
from naxi.v_0d1.gridman.dataloader import StreamLoader
from naxi.v_0d1.gridman.tools import save_checkpoint, load_checkpoint, print_model_parameters

from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup


def reduce_value(value: torch.Tensor):
    if not dist.is_initialized():
        return value
    val = value.data.clone()
    # 求和
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    # 真实平均值
    return val / dist.get_world_size()


def train_model(is_sft: bool = False):
    config = RUNNING_CONFIG
    dtype = torch.bfloat16

    # 初始化多卡环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if is_sft:
        mode_name = 'SFT'
        model_name = f'{config.name}_sft'
        dataset_file = config.sft_file
        chunk_size = config.sft_chunk_size
        bptt_size = config.sft_bptt_size
        lr = config.sft_lr
        steps = config.sft_steps
    else:
        mode_name = 'PRE-TRAIN'
        model_name = f'{config.name}_pretrain'
        dataset_file = config.pretrain_train_file
        chunk_size = config.pretrain_chunk_size
        bptt_size = config.pretrain_bptt_size
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

    tokenizer = config.tokenizer
    grid_man = Gridman(config).to(device)
    writer = None

    if local_rank == 0:
        print(f'🚀 正在初始化 {mode_name}...')
        print_model_parameters(grid_man)
        print('\n' + '>'*25 + f' 开始极速流式 {mode_name} ' + '<'*25)

        log_dir = os.path.join('log', model_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f'📊 TensorBoard 日志将保存至: {log_dir}')

    if is_sft:
        # 加载预训练模型
        load_checkpoint(grid_man, is_sft, need_print=(local_rank==0))

    # 此处为强制类型标记
    grid_man: Gridman = DDP(grid_man, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    grid_man_module: Gridman = grid_man.module

    optimizer = torch.optim.AdamW(grid_man.parameters(), lr=lr)
    total_update_steps = steps // bptt_size

    num_warmup_steps = int(total_update_steps * 0.05) 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_update_steps
    )

    loss_acc = torch.tensor(0.0, device=device)
    loss_acc_log = torch.tensor(0.0, device=device)

    for step in range(steps): 
        grid_man.train()
        
        # 接收 Token 和 Mask
        input_patches, mask_patches = dataloader.get_batch()
        input_patches = input_patches.to(device)
        mask_patches = mask_patches.to(device)
        
        # 构造 Input 和 Target
        inputs = input_patches[:, :-1]   
        targets = input_patches[:, 1:].clone()  # 防止修改原 tensor
        target_masks = mask_patches[:, 1:]      # 与 targets 对应
        
        targets[target_masks == 0] = -100

        optimizer.zero_grad()

        # 最后一次 forward 触发同步
        is_update_step = ((step + 1) % bptt_size == 0)
        sync_context = grid_man.no_sync() if not is_update_step else nullcontext()
        
        with sync_context:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = grid_man(inputs)
                grid_man_module.core_ouro.mem_sync()

                if (targets != -100).any():
                    batch_size, length = targets.shape

                    # 计算交叉熵损失
                    base_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        targets.reshape(-1),
                        ignore_index=-100  # 忽略 -100 
                    )

                    if is_sft:
                        # 余弦衰减, 靠前损失权重越大, 强化记忆
                        alpha = 1.618  
                        positions = torch.arange(length, device=device, dtype=torch.float32)
            
                        weights_1d = 1.0 + alpha * (1.0 + torch.cos(torch.pi * positions / (length-1))) / 2.0
                        weights_1d = weights_1d.to(logits.dtype)
                        
                        weights = weights_1d.unsqueeze(0).expand(batch_size, length).reshape(-1)
                        
                        valid_mask = (targets.reshape(-1) != -100)
                        valid_weights = weights * valid_mask
                        
                        loss = (base_loss * valid_weights).sum() / (valid_weights.sum() + 1e-8)
                    else:
                        loss = base_loss

                else: # 整个 patch -100 是 0 loss
                    loss = logits.sum() * 0.0
        
        loss_acc = loss_acc + loss

        with torch.no_grad():
            dist_loss = reduce_value(loss)
            loss_acc_log = loss_acc_log + dist_loss

        if is_update_step:
            loss_acc = loss_acc / bptt_size
            loss_acc.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(grid_man.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            grid_man_module.core_ouro.mem_detach()

            if local_rank == 0:
                avg_loss = loss_acc_log.item() / bptt_size
                writer.add_scalar('Train/Loss', avg_loss, step)
                writer.add_scalar('Train/Grad_Norm', total_norm, step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)

                print(f'\n📌 Step {step+1} | Total Loss = {loss_acc_log.item()/bptt_size:.4f}')

            loss_acc = torch.tensor(0.0, device=device)
            loss_acc_log = torch.tensor(0.0, device=device)

        if (step+1) % 1000 == 0 and local_rank == 0:
            save_checkpoint(grid_man_module, model_name)
            
            grid_man.eval()
            if is_sft:
                prompt_ids_list = (
                    [tokenizer.user_token_id] + 
                    tokenizer.encode('你有自我意识吗') + 
                    [tokenizer.eos_token_id, tokenizer.assistant_token_id]
                )
            else:
                prompt_ids_list = tokenizer.encode('今天天气还')
        
            prompt_ids = torch.tensor([[tokenizer.eos_token_id] + prompt_ids_list], dtype=torch.long, device=device)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                generated_ids = grid_man_module.generate(prompt_ids)
            
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            print(f'Gridman 🤖: {generated_text}')
            print('-' * 65)


def generate_test(is_sft: False):
    """仅进行生成测试"""
    config = RUNNING_CONFIG
    device = config.device
    tokenizer = config.tokenizer
    dtype = config.dtype

    grid_man = Gridman(config).to(device)
    load_checkpoint(grid_man, is_sft)
    grid_man.eval()
    if is_sft:
        prompt_ids_list = (
            [tokenizer.eos_token_id, tokenizer.user_token_id] + 
            tokenizer.encode('你是谁') + 
            [tokenizer.eos_token_id, tokenizer.assistant_token_id]
        )
    else:
        prompt_ids_list = tokenizer.encode('世界上最高的山是')

    prompt_ids = torch.tensor([[tokenizer.eos_token_id] + prompt_ids_list], dtype=torch.long, device=device)
    
    with torch.amp.autocast(config.device_type, dtype=dtype):
        generated_ids = grid_man.generate(prompt_ids)
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f'Gridman 🤖: {generated_text}')
    print('-' * 65)