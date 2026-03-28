import torch
import torch.nn as nn

from ouro_core import Ouro
from config import Config
    

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
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 0.7) -> torch.Tensor:
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