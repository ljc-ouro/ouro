import os
import torch
import torch.nn as nn
from naxi.v_0d1.gridman.config import Config, RUNNING_CONFIG


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
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"Gridman 🤖 模型体积统计:")
    print(f" ├─ 总规模: {total_tracked / 1e6:.2f} M")
    print(f" ├─ 可训练参数: {trainable_params / 1e6:.2f} M")
    print(f" └─ 记忆状态量: {mem_buffers_size / 1e6:.2f} M")
    print("="*60)


def save_checkpoint(model: nn.Module, is_sft: bool = False, config: Config = RUNNING_CONFIG):
    if is_sft:
        model_name = f'{config.name}_{config.version}_sft'
    else:
        model_name = f'{config.name}_{config.version}_pretrain'

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    checkpoint_path = os.path.join(config.checkpoint_dir, f'{model_name}.pt')
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: nn.Module, is_sft: bool = False, need_print: bool = True, config: Config = RUNNING_CONFIG) -> bool:
    if is_sft:
        model_name = f'{config.name}_{config.version}_sft'
    else:
        model_name = f'{config.name}_{config.version}_pretrain'

    checkpoint_path = os.path.join(config.checkpoint_dir, f'{model_name}.pt')
    
    if not os.path.exists(checkpoint_path):
        print(checkpoint_path)
        raise FileNotFoundError(f'⚠️ 未找到检查点: {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if need_print: 
        print(f'🔄 正在从 {checkpoint_path} 加载模型权重...')
        print('✅ 模型权重加载成功!')
