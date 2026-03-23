from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from ouro_core import Ouro, ByteTokenizer
    

@dataclass
class Config:
    embed_dim: int = 768
    block_layers: int = 8
    blocks: int = 2

    patch_size: int = 768
    chunk_size: int = 14
    bptt_size: int = 4

    # 预训练配置
    pretrain_train_file: str = f'./datasets/pretrain_hq/pretrain_hq_train.jsonl'
    tokenizer: ByteTokenizer = ByteTokenizer()


class StreamLoader:
    """
    流式数据加载器, 多端点并行流读取
    """
    def __init__(self, config: Config):
        self.config = config
        self.chunk_size = config.chunk_size
        self.patch_size = config.patch_size + 1
        self.tokenizer = config.tokenizer
        
        print(f"📦 初始化流式数据加载: {config.pretrain_train_file}...")
        
        # 为每个 chunk 维护 buffer 和迭代器
        self.buffers = [[] for _ in range(self.chunk_size)]
        self.iterators = []
        
        # 计算文件等分点
        file_size = os.path.getsize(config.pretrain_train_file)
        step_size = file_size // self.chunk_size
        
        for i in range(self.chunk_size):
            start_offset = i * step_size
            self.iterators.append(self._get_stream(start_offset))

    def _get_stream(self, start_offset: int):
        while True: 
            with open(self.config.pretrain_train_file, 'rb') as f:
                f.seek(start_offset)
                if start_offset > 0:
                    f.readline() 
                
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        decoded_line = line.decode('utf-8')
                        example = json.loads(decoded_line)
                        text = example.get('text', '')
                        if text:
                            text = text.replace('<|im_end|>', '')
                            yield from self.tokenizer.encode(text)
                            yield self.tokenizer.eos_token_id
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
            
            # 末尾从头循环
            start_offset = 0

    def get_batch(self) -> torch.Tensor:
        # 从 n 个流中分别读取 patch_size 个 token
        for i in range(self.chunk_size):
            while len(self.buffers[i]) < self.patch_size:
                self.buffers[i].append(next(self.iterators[i]))
        
        batch_bytes = []
        for i in range(self.chunk_size):
            batch_bytes.append(self.buffers[i][:self.patch_size])
            self.buffers[i] = self.buffers[i][self.patch_size:]

        # [chunk_size, patch_size]
        input_patches = torch.tensor(batch_bytes, dtype=torch.long)
        return input_patches
    

class Gridman(nn.Module):
    """
    基于 Ouro 架构的自回归语言模型 Gridman
    """
    def __init__(self, config = Config()):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
        # 嵌入层
        self.byte_emb = nn.Embedding(config.tokenizer.vocab_size, config.embed_dim)
        
        # Ouro
        self.core_ouro = Ouro(self.embed_dim, config.blocks, config.block_layers)
        
        # 输出头
        self.out_proj = nn.Linear(self.embed_dim, config.tokenizer.vocab_size)

        torch.nn.init.normal_(self.byte_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.1, std=0.02)
        torch.nn.init.zeros_(self.out_proj.bias)

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.byte_emb(x)
        x = self.core_ouro(x)
        logits = self.out_proj(x)
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 0.8) -> torch.Tensor:
        """
        自回归推理生成
        """
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # 采样
            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 遇到 EOS 停止生成
            if next_token.item() == self.config.tokenizer.eos_token_id:
                break
                
        return input_ids


def print_model_parameters(model: nn.Module):
    """打印模型参数统计信息"""
    total_params = 0
    trainable_params = 0
    
    # 遍历所有参数
    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    # 打印汇总信息
    print("\n" + "="*60)
    print(f"Gridman 🤖 参数统计: {(total_params)/1e6:.2f} M")
    print("="*60 + "\n")


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


if __name__ == "__main__":
    # 开启时只进行生成测试
    test_only= False
    config = Config()
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if test_only:
        grid_man = Gridman(config).to(device)
        load_checkpoint(grid_man, device=device)
        grid_man.eval()
        prompt_text = '北京是一个'
        
        prompt_ids = torch.tensor([config.tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
        
        with torch.amp.autocast('cuda', dtype=dtype):
            generated_ids = grid_man.generate(prompt_ids)
        
        generated_text = config.tokenizer.decode(generated_ids[0].tolist())
        print(f"Gridman 🤖:: {generated_text}")
        print("-" * 65)

    else:
        # 标准预训练
        print("🚀 正在初始化预训练...")
        grid_man = Gridman(config).to(device)
        
        optimizer = torch.optim.AdamW(grid_man.parameters(), lr=3e-4)
        print_model_parameters(grid_man)
        
        dataloader = StreamLoader(config)
        tokenizer = config.tokenizer
        
        print("\n" + ">"*25 + " 开始极速流式预训练 " + "<"*25)

        loss_acc = torch.tensor(0.0, device=device)

        for step in range(1200000): 
            grid_man.train()
            
            input_patches = dataloader.get_batch().to(device)
            
            # 构造 Input 和 Target
            inputs = input_patches[:, :-1]   
            targets = input_patches[:, 1:]  

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = grid_man(inputs)
                # 计算交叉熵损失
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    targets.reshape(-1)
                )
            
            loss_acc = loss_acc + loss

            if step % config.bptt_size == 0:
                loss_acc = loss_acc / config.bptt_size
                loss_acc.backward()
                torch.nn.utils.clip_grad_norm_(grid_man.parameters(), 1.0)
                optimizer.step()
                grid_man.core_ouro.mem_detach()
                print(f"\n📌 Step {step+1} | Total Loss = {loss_acc.item():.4f}")
                loss_acc = torch.tensor(0.0, device=device)

            if (step+1) % 1000 == 0:
                save_checkpoint(grid_man)
                
                grid_man.eval()
                prompt_text = "人工智能是一种"
            
                prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
                
                with torch.amp.autocast('cuda', dtype=dtype):
                    generated_ids = grid_man.generate(prompt_ids)
                
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                print(f"Gridman 🤖: {generated_text}")
                print("-" * 65)