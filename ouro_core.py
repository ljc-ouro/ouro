import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ByteTokenizer:
    """
    字节 Tokenizer, 无需训练, 全世界通用
    """
    def __init__(self):
        # 填充符
        self.pad_token_id = 256 
        # 终止符
        self.eos_token_id = 257

        self.vocab_size = self.eos_token_id + 1

    def encode(self, text: str) -> list[int]:
        # 将文本转为 UTF-8 字节列表
        return list(text.encode('utf-8'))

    def decode(self, ids: list[int]) -> str:
        # 过滤特殊 token 并解码
        clean_ids = [i for i in ids if 0 <= i < 256]
        return bytes(clean_ids).decode('utf-8', errors='replace')
    
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


class OuroLayer(nn.Module):
    """
    模块结构:
    - Pre_Norm
    - Pre_FFN
    - 前缀注意力
    - 动态 FFN 层: LayerNorm(因果 Mem @ Linear)
    - 残差连接
    """
    def __init__(self, embed_dim: int, need_mem: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = self.embed_dim // 64
        self.head_dim = self.embed_dim // self.num_heads
        self.need_mem = need_mem

        self.act = nn.SiLU()
        self.ffn_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4 * 11),
            self.act,
            nn.Linear(embed_dim // 4 * 11, embed_dim)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head_norm = nn.LayerNorm(self.head_dim)

        # 开启标准的 Delte Rule 实现
        if self.need_mem:
            # 全局记忆矩阵 
            self.mem: torch.Tensor
            self.register_buffer('mem', torch.zeros(1, self.embed_dim, self.embed_dim))
            self.mem_g = nn.Linear(embed_dim, embed_dim)

            self.mem_norm = nn.LayerNorm(embed_dim)

            self.w_q = nn.Linear(embed_dim, embed_dim)
            self.w_k = nn.Linear(embed_dim, embed_dim)
            self.w_v = nn.Linear(embed_dim, embed_dim)
            self.w_g = nn.Linear(embed_dim, embed_dim)
            self.w_o = nn.Linear(embed_dim, embed_dim * 4)

        self.o_proj = nn.Linear(embed_dim * 4 if self.need_mem else embed_dim, embed_dim)
        self.context_norm = nn.LayerNorm(self.embed_dim)

        torch.nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.o_proj.bias)

    def __call__(self, x: torch.Tensor, seq_dim: int = 1):
        return self.forward(x, seq_dim)
    
    def mem_detach(self):
        if self.need_mem:
            self.mem = self.mem.detach()

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor | None]:
        if seq_dim < 0:
            seq_dim = x.dim() + seq_dim

        """
        前缀注意力
        """
        x_norm: torch.Tensor = self.norm(x)
        x_mlp: torch.Tensor = x + self.ffn_layer(x_norm)

        # 计算前缀和
        x_cumsum = torch.cumsum(x_mlp, dim=seq_dim)
        init_shape = x_mlp.shape[:-1]

        # 多头切分
        x_h = x_mlp.view(*init_shape, self.num_heads, self.head_dim)
        xc_h = x_cumsum.view(*init_shape, self.num_heads, self.head_dim)
        
        # 相似性 Gate 
        alignment: torch.Tensor = torch.sum(self.head_norm(x_h) * self.head_norm(xc_h), dim=-1, keepdim=True) / (self.head_dim ** 0.5)
        gate = torch.sigmoid(alignment)
        
        # 计算加权的序列
        weighted_x_h = x_h * gate  
        
        # 重新展平, 加权序列 [gate_0*x_0, gate_1*x_1, ...]
        weighted_x = weighted_x_h.reshape(*init_shape, self.embed_dim)
        context = torch.cumsum(weighted_x, dim=seq_dim)

        """
        动态 FFN
        """
        # 构建动态的 FFN 层 [prev_mem * decay + cumsum(k^T @ v_dyn) / (self.embed_dim ** 0.5)] @ w_o
        if self.need_mem:
            context: torch.Tensor = self.mem_norm(x_mlp + context)

            context_q: torch.Tensor = self.w_q(context)
            context_k: torch.Tensor = self.w_k(context)
            context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5) 
            context_v: torch.Tensor = self.w_v(context)
            context_g = torch.sigmoid(self.w_g(context))
        
            batch_size, seq_len, _ = x.shape

            prev_mem = self.mem.expand(batch_size, -1, -1).to(x.device)

            # 预测的 V
            v_retrieved: torch.Tensor = torch.bmm(context_k, prev_mem) / (self.embed_dim ** 0.5)

            # 计算真实 V 与预测 V 的 Delta
            delta_v = context_v - v_retrieved
            v_dyn = context_g * delta_v
            
            # 外积更新
            delta_mem: torch.Tensor = torch.bmm(context_k.transpose(-1, -2), v_dyn) / (self.embed_dim ** 0.5)
            
            # 记忆更新
            mem_g: torch.Tensor = self.act(self.mem_g(prev_mem + delta_mem))
            mem_g = torch.sigmoid(mem_g)
            next_mem: torch.Tensor = (mem_g * prev_mem + delta_mem)

            self.mem = next_mem.mean(0, keepdim=True)

            # q_t @ [prev_mem * decay + cumsum(k^T @ v_dyn) / (self.embed_dim ** 0.5)]
            # (q_t @ prev_mem) + (q_t @ k^T) @ v_dyn
            
            # 历史记忆 
            mem_out_prev = torch.bmm(context_q, prev_mem)
            
            # Q 与 K 的标准注意力打分矩阵 (Q @ K^T) -> [B, L, L]
            scores = torch.bmm(context_q, context_k.transpose(-1, -2))
            
            # 因果掩码 
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=scores.dtype))
            scores_causal = scores * causal_mask
           
            # 标准注意力 (scores_causal @ V_dyn) -> [B, L, D]
            mem_out_delta: torch.Tensor = torch.bmm(scores_causal, v_dyn) / (self.embed_dim ** 0.5)
            
            # 合并输出
            mem_out: torch.Tensor = (mem_out_prev + mem_out_delta) / (self.embed_dim ** 0.5)

            context_o: torch.Tensor = self.w_o(mem_out)
            context = self.act(context_o)
        
        context = self.o_proj(context)
        context = self.context_norm(context)

        return x_mlp + context, None if not self.need_mem else scores_causal
    
    
class OuroBlock(nn.Module):
    """
    OuroLayer 堆叠的模块

    结构与标准 Attention 一致:
    - Pre_Norm
    - Attention
    - FFN
    - 残差连接

    Attention 由 OuroLayer 行为涌现
    """
    def __init__(self, embed_dim: int, block_layers: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.block_layers = block_layers

        self.act = nn.SiLU()

        self.ouro_self_attn_proj = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.ouro_self_attn_norm = nn.LayerNorm(self.embed_dim)

        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.ouro_layers: nn.ModuleList[OuroLayer] = nn.ModuleList([
            OuroLayer(self.embed_dim, self.is_prime(_+1)) for _ in range(self.block_layers)
        ])

        self.ffn_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            self.act,
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, x: torch.Tensor, seq_dim: int = 1):
        return self.forward(x, seq_dim)

    def mem_detach(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_detach()

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        residual = x

        ouro_self_attn = torch.tensor(0.0)

        for layer in self.ouro_layers:
            layer: OuroLayer
            if layer.need_mem:
                x, attn = layer(x, seq_dim)
                ouro_self_attn = ouro_self_attn + attn
            else:
                x: torch.Tensor
                x, _ = checkpoint(layer, x, seq_dim, use_reentrant=False)

        batch_size, seq_len, _ = x.shape

        # 涌现注意力 (Emergent Attention)
        ouro_self_attn_residual = ouro_self_attn
        ouro_self_attn = torch.bmm(ouro_self_attn, x)
        ouro_self_attn_proj = self.ouro_self_attn_proj.unsqueeze(0).expand(batch_size, -1, -1)
        ouro_self_attn: torch.Tensor = torch.bmm(ouro_self_attn, ouro_self_attn_proj) / (self.embed_dim ** 0.5)
        ouro_self_attn = self.act(ouro_self_attn)
        ouro_self_attn_normed: torch.Tensor = self.ouro_self_attn_norm(ouro_self_attn)
        ouro_self_attn = torch.bmm(ouro_self_attn_normed, ouro_self_attn_normed.transpose(-1, -2)) 
        ouro_self_attn = (ouro_self_attn_residual + ouro_self_attn) / (self.embed_dim ** 0.5)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=ouro_self_attn.dtype))
        ouro_self_attn = ouro_self_attn.masked_fill(causal_mask == 0, float('-inf'))
        ouro_self_attn = torch.softmax(ouro_self_attn, dim=-1) 

        residual = residual + torch.bmm(ouro_self_attn, self.w_v(self.v_norm(residual)))
        x = self.ffn_layer(self.norm(residual))

        x = residual + x
        return x
    

class Ouro(nn.Module):
    """
    Ouro 标准模型
    由 OuroBlocks 堆叠而成
    在 OuroBlocks 尺度下 Ouro 退化成标准的 Transformer
    """
    def __init__(self, embed_dim: int, blocks: int, block_layers: int = 8):
        super().__init__()

        self.ouro_blocks = nn.ModuleList([
            OuroBlock(embed_dim, block_layers) for _ in range(blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: torch.Tensor, seq_dim: int = 1):
        return self.forward(x, seq_dim)

    def mem_detach(self):
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_detach()

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        for block in self.ouro_blocks:
            x = block(x, seq_dim)
        return self.norm(x)

        