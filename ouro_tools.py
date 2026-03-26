import os
import torch
import torch.nn as nn


def print_model_parameters(model: nn.Module):
    """打印模型参数和状态统计信息"""
    total_params = 0
    trainable_params = 0
    total_buffers = 0
    
    # 标准参数
    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    # Buffer
    for _, buffer in model.named_buffers():
        total_buffers += buffer.numel()
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"🤖 Gridman 模型体积统计:")
    print(f" ├─ 总参数量 (Parameters):  {(total_params + total_buffers) / 1e6:.2f} M")
    print(f" ├─ 可训练参数 (Trainable): {trainable_params / 1e6:.2f} M")
    print(f" └─ 记忆状态量 (Buffers):   {total_buffers / 1e6:.2f} M")
    print("="*60)


def save_checkpoint(model: nn.Module, save_dir: str = '.'):
    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    
    checkpoint_path = os.path.join(save_dir, 'model.pt')
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: nn.Module, load_dir: str = '.', device: torch.device | str = 'cpu') -> bool:
    checkpoint_path = os.path.join(load_dir, 'model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 未找到检查点: {checkpoint_path}")
        return False
        
    print(f"🔄 正在从 {checkpoint_path} 加载模型权重...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✅ 模型权重加载成功！")
    return True