import torch
import torch.nn as nn

from naxi.v_0d2.ouro.core import Ouro
from naxi.v_0d2.gridman.config import Config


class Gridman(nn.Module):
    """
    基于 Ouro 架构的自回归语言模型 Gridman
    """
    def __init__(self, config=Config()):
        super().__init__()

        self.config = config
        self.embed_dim = config.embed_dim

        # 嵌入层
        self.byte_emb = nn.Embedding(config.tokenizer.vocab_size, config.embed_dim)

        # Ouro
        self.core_ouro = Ouro(self.embed_dim, config.chunk_size, config.blocks, config.block_layers)

        # 输出头
        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, config.tokenizer.vocab_size)

        torch.nn.init.normal_(self.byte_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.out_proj.weight, mean=0.1, std=0.02)
        torch.nn.init.zeros_(self.out_proj.bias)

    # def __call__(self, x: torch.Tensor, lock_mem: bool = False):
    #     return self.forward(x, lock_mem)

    def forward(self, x: torch.Tensor, lock_mem: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.byte_emb(x)

        x = self.core_ouro(x, lock_mem)
        ar_logits = self.out_proj(self.out_norm(x))
        return ar_logits