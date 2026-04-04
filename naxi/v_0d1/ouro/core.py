import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
    

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, head_dim: int, max_seq_len=4096, base=50000.0):
        super().__init__()

        self.inv_freq: torch.Tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        self.max_seq_len = max_seq_len
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self._build_cache(max_seq_len)

    def __call__(self, seq_len: int):
        return self.forward(seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]


def rotate_half(x: torch.Tensor):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = embed_dim // 64
        self.head_dim = embed_dim // self.heads

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        qkv: torch.Tensor = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(y)


class OuroAttentionMixer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.att_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4 * 6),
            nn.SiLU(),
            nn.Linear(embed_dim // 4 * 6, embed_dim)
        )

        self.attn = CausalSelfAttention(embed_dim)

    def __call__(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return self.forward(x, cos, sin)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x_ffn = x + self.ffn(self.norm(x))
        attn_context = self.attn(self.att_norm(x_ffn), cos, sin)
        return attn_context, x_ffn


class OuroLayer(nn.Module):
    """
    模块结构:
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

        self.attn_mixer = OuroAttentionMixer(self.embed_dim)

        self.act = nn.SiLU()

        # 开启标准的 Delte Rule 实现
        if self.need_mem:
            self._pending_mem = None

            # 全局记忆矩阵 
            self.mem: torch.Tensor
            self.register_buffer('mem', torch.eye(self.embed_dim).unsqueeze(0))
            self.mem_g = nn.Linear(embed_dim, embed_dim)

            self.mem_norm = nn.LayerNorm(embed_dim)

            self.w_q = nn.Linear(embed_dim, embed_dim)
            self.w_k = nn.Linear(embed_dim, embed_dim)
            self.w_v = nn.Linear(embed_dim, embed_dim)
            self.w_g = nn.Linear(embed_dim, embed_dim)
            self.w_o = nn.Linear(embed_dim, embed_dim * 4)

            self.o_proj = nn.Linear(embed_dim * 4 if self.need_mem else embed_dim, embed_dim)

            torch.nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.o_proj.bias)

    def __call__(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lock_mem: bool = False):
        return self.forward(x, cos, sin, lock_mem)
    
    def mem_detach(self):
        if self.need_mem:
            self.mem = self.mem.detach()

    def mem_clear(self):
        if self.need_mem:
            self.mem = torch.zeros(1, self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lock_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        注意力混合
        """
        attn_context, x_ffn = self.attn_mixer(x, cos, sin)

        """
        动态 FFN
        """
        # 构建动态的 FFN 层 [prev_mem * decay + cumsum(k^T @ v_dyn) / (self.embed_dim ** 0.5)] @ w_o
        if self.need_mem:
            mem_context: torch.Tensor = self.mem_norm(x_ffn + attn_context)

            context_q: torch.Tensor = self.w_q(mem_context)
            context_q = self.act(context_q)
            context_k: torch.Tensor = self.w_k(mem_context)
            context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5) 
            context_v: torch.Tensor = self.w_v(mem_context)
            context_g = torch.sigmoid(self.w_g(mem_context))
        
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

            next_mem: torch.Tensor = mem_g * prev_mem + delta_mem

            if not lock_mem:
                self._pending_mem = next_mem.mean(0, keepdim=True)

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
            mem_out_delta: torch.Tensor = torch.bmm(scores_causal, v_dyn) 
            
            # 合并输出
            mem_out: torch.Tensor = (mem_out_prev + mem_out_delta) / (self.embed_dim ** 0.5)

            context_o: torch.Tensor = self.w_o(mem_out)
            mem_context = self.act(context_o)
        
            mem_context = self.o_proj(mem_context)

            return x_ffn + attn_context + mem_context, scores_causal

        return x_ffn + attn_context, None
    
    
class OuroBlock(nn.Module):
    """
    OuroLayer 堆叠的模块

    模块结构:
    - Pre_Norm
    - Attention
    - FFN
    - 残差连接

    Attention 由 OuroLayer 行为涌现
    """
    def __init__(self, embed_dim: int, rope: RotaryPositionalEmbeddings, block_layers: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.rope = rope
        self.block_layers = block_layers

        self.act = nn.SiLU()

        self.ouro_self_attn_proj = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.ouro_self_attn_norm = nn.LayerNorm(self.embed_dim)

        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.ouro_self_attn_gate = nn.Linear(embed_dim, embed_dim, bias=False)

        self.ouro_layers: nn.ModuleList[OuroLayer] = nn.ModuleList([
            OuroLayer(self.embed_dim, self.is_prime(_+1)) for _ in range(self.block_layers)
        ])

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            self.act,
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lock_mem: bool = False):
        return self.forward(x, cos, sin, lock_mem)

    def mem_detach(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_detach()

    def mem_clear(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_clear()

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x

        ouro_self_attn = torch.tensor(0.0)

        for layer in self.ouro_layers:
            layer: OuroLayer
            if layer.need_mem:
                x, attn = layer(x, cos, sin, lock_mem)
                ouro_self_attn = ouro_self_attn + attn
            else:
                x: torch.Tensor
                x, _ = checkpoint(layer, x, cos, sin, use_reentrant=False)

            inner_residual = x

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

        # 注意力门控
        ouro_self_attn_output = torch.bmm(ouro_self_attn, self.w_v(self.v_norm(residual))) / (self.embed_dim ** 0.5)
        gate = torch.sigmoid(self.act(self.ouro_self_attn_gate(residual)))
        residual = residual + gate * ouro_self_attn_output

        # 标准输出
        x = self.ffn(self.norm(residual))
        x = x + residual + inner_residual
        return x
    

class Ouro(nn.Module):
    """
    Ouro 标准模型
    由 OuroBlocks 堆叠而成
    """
    def __init__(self, embed_dim: int, blocks: int, block_layers: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = blocks

        self.rope = RotaryPositionalEmbeddings(64)

        self.ouro_blocks = nn.ModuleList([
            OuroBlock(embed_dim, self.rope, block_layers) for _ in range(blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.attnres_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(embed_dim)) for _ in range(self.blocks)
        ])
        
        self.attnres_k_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def __call__(self, x: torch.Tensor, lock_mem: bool = False):
        return self.forward(x, lock_mem)

    def mem_detach(self):
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_detach()

    def mem_clear(self):
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_clear()

    def mem_sync(self):
        """
        同步所有 OuroLayer 的 mem
        """
        pending_mems = []
        mem_layers = []

        # 收集所有需要同步更新的 mem
        for block in self.ouro_blocks:
            block: OuroBlock
            for layer in block.ouro_layers:
                layer: OuroLayer
                if layer.need_mem and layer._pending_mem is not None:
                    pending_mems.append(layer._pending_mem)
                    mem_layers.append(layer)

        if dist.is_initialized():
            stacked_mems = torch.stack(pending_mems, dim=0)
            
            with torch.no_grad():
                dist.all_reduce(stacked_mems, op=dist.ReduceOp.SUM)
                stacked_mems = stacked_mems / dist.get_world_size()

            # 将同步后的 mem 更新回各个 layer
            for i, layer in enumerate(mem_layers):
                local_mem: torch.Tensor = pending_mems[i]
                synced_mem = stacked_mems[i]
                layer.mem = local_mem + (synced_mem - local_mem).detach()
                layer._pending_mem = None
        else:
            # 单卡环境
            for layer in mem_layers:
                layer.mem = layer._pending_mem
                layer._pending_mem = None            

    def forward(self, x: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        _, seq_len, _ = x.shape
        cos, sin = self.rope(seq_len)

        history_states = [x]

        # 注意力残差
        for i, block in enumerate(self.ouro_blocks):
            if i > 0:
                stacked_history = torch.stack(history_states, dim=2)
                
                keys = self.attnres_k_norm(stacked_history) 
                values = stacked_history 
                q = self.attnres_queries[i - 1] 

                scores = torch.matmul(keys, q) / (self.embed_dim ** 0.5)
                alpha = F.softmax(scores, dim=-1)
                x = torch.sum(alpha.unsqueeze(-1) * values, dim=2)

            x = block(x, cos, sin, lock_mem)
            history_states.append(x)

            residual = x

        stacked_history = torch.stack(history_states, dim=2)
        
        keys = self.attnres_k_norm(stacked_history) 
        values = stacked_history 
        q = self.attnres_queries[-1]

        scores = torch.matmul(keys, q) / (self.embed_dim ** 0.5)
        alpha = F.softmax(scores, dim=-1)
        x = torch.sum(alpha.unsqueeze(-1) * values, dim=2)

        x = residual + self.ffn(x)

        return self.norm(x)

        


