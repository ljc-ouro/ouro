import torch
import torch.nn.functional as F

from config import RUNNING_CONFIG
from grid_man import Gridman
from ouro_dataloader import StreamLoader
from ouro_tools import save_checkpoint, load_checkpoint, print_model_parameters


def train_model(is_sft: bool = False):
    config = RUNNING_CONFIG
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_sft:
        mode_name = 'SFT'
        dataset_file = config.sft_file
        chunk_size = config.sft_chunk_size
        bptt_size = config.sft_bptt_size
        lr = config.sft_lr
        steps = config.sft_steps
    else:
        mode_name = 'PRE-TRAIN'
        dataset_file = config.pretrain_train_file
        chunk_size = config.pretrain_chunk_size
        bptt_size = config.pretrain_bptt_size
        lr = config.pretrain_lr
        steps = config.pretrain_steps

    dataloader = StreamLoader(
        patch_size=config.patch_size, 
        chunk_size=chunk_size, 
        datasets=dataset_file,
        is_sft=is_sft
    )

    tokenizer = config.tokenizer

    print(f'🚀 正在初始化 {mode_name}...')
    grid_man = Gridman(config).to(device)
    print_model_parameters(grid_man)

    if is_sft:
        # 加载预训练模型
        load_checkpoint(grid_man)

    optimizer = torch.optim.AdamW(grid_man.parameters(), lr=lr)
    
    print('\n' + '>'*25 + f' 开始极速流式 {mode_name} ' + '<'*25)

    loss_acc = torch.tensor(0.0, device=device)

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
        
        with torch.amp.autocast('cuda', dtype=dtype):
            logits = grid_man(inputs)

            if (targets != -100).any():
                # 计算交叉熵损失
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    targets.reshape(-1),
                    ignore_index=-100  # 忽略 -100 
                )
            else: # 整个 patch -100 是 0 loss
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss_acc = loss_acc + loss

        if step % bptt_size == 0:
            loss_acc = loss_acc / bptt_size
            loss_acc.backward()
            torch.nn.utils.clip_grad_norm_(grid_man.parameters(), 1.0)
            optimizer.step()
            grid_man.core_ouro.mem_detach()
            print(f'\n📌 Step {step+1} | Total Loss = {loss_acc.item():.4f}')
            loss_acc = torch.tensor(0.0, device=device)

        if (step+1) % 5000 == 0:
            save_checkpoint(grid_man)
            
            grid_man.eval()
            if is_sft:
                prompt_ids_list = (
                    [tokenizer.user_token_id] + 
                    tokenizer.encode('你好') + 
                    [tokenizer.eos_token_id, tokenizer.assistant_token_id]
                )
            else:
                prompt_ids_list = tokenizer.encode('今天天气还')
        
            prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                generated_ids = grid_man.generate(prompt_ids)
            
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            print(f'Gridman 🤖: {generated_text}')
            print('-' * 65)


def generate_test():
    """仅进行生成测试"""
    config = RUNNING_CONFIG
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    grid_man = Gridman(config).to(device)
    load_checkpoint(grid_man, device=device)
    grid_man.eval()
    prompt_text = '中国主席是'
    
    prompt_ids = torch.tensor([config.tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    
    with torch.amp.autocast('cuda', dtype=dtype):
        generated_ids = grid_man.generate(prompt_ids)
    
    generated_text = config.tokenizer.decode(generated_ids[0].tolist())
    print(f'Gridman 🤖: {generated_text}')
    print('-' * 65)


if __name__ == '__main__':
    test_only = False
    is_sft = False

    if test_only:
        generate_test()
    else:
        train_model(is_sft=is_sft)