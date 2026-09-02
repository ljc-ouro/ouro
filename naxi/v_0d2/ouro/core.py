import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


HEAD_DIM = 64


class OuroNorm(nn.Module):
    def __init__(self, embed_dim: int, init_bias: float = 0):
        super().__init__()
        self.embed_dim = embed_dim
    
        self.k_proj = nn.Linear(embed_dim, 1)
        self.act = nn.Sigmoid()

        nn.init.zeros_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, init_bias)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.act(self.k_proj(x))
        x_normed = F.normalize(x, p=2.0, dim=-1) * (self.embed_dim ** 0.5)
        return k * x_normed
    

class OuroCell(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
    ):
        super().__init__()
        self.intrinsic_loss = 0.0
        self.linear = nn.Linear(in_features, out_features, bias)

        self.multiple_of = 16
        hidden_dim = int(out_features / 16)
        self.rank = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank)
        )

        self._pending_mem = None

        # 全局记忆矩阵 
        self.mem: torch.Tensor
        self.register_buffer('mem', torch.eye(self.rank).unsqueeze(0))
        self.mem_norm = nn.LayerNorm(self.rank)
        self.w_qkvgd = nn.Linear(self.rank, self.rank * 5)
        self.act = nn.SiLU()
        self.hidden_norm = nn.LayerNorm(self.rank)
        self.w_o = nn.Linear(self.rank, self.rank, bias=False)
    
        with torch.no_grad():
            torch.nn.init.normal_(
                self.w_qkvgd.weight[3 * self.rank : 4 * self.rank, :], 
                mean=0.0, std=0.02
            )
            torch.nn.init.constant_(
                self.w_qkvgd.bias[3 * self.rank : 4 * self.rank], 
                -6.0
            )

            nn.init.kaiming_uniform_(
                self.lora_A,
                a=math.sqrt(5),
            )

            nn.init.normal_(
            self.lora_B,
                mean=0.0,
                std=1e-2,
            )

    def mem_detach(self):
        self.mem = self.mem.detach()

    def mem_clear(self):
        self.mem = torch.zeros(1, self.rank, self.rank)

    def forward(self, x: torch.Tensor, lock_mem: bool = False):
        orig_shape = x.shape
        
        if len(orig_shape) == 2:
            x = x.unsqueeze(1)  
        elif len(orig_shape) > 3:
            x = x.contiguous().view(-1, orig_shape[-2], orig_shape[-1])
            
        _, seq_len, _ = x.shape

        base_output = self.linear(x)

        x_low_rank = x @ self.lora_A.T

        qkvgd: torch.Tensor = self.w_qkvgd(self.mem_norm(x_low_rank))
        context_q, context_k, context_v, context_g, context_d = qkvgd.chunk(5, dim=-1)

        context_q = F.normalize(context_q, p=2, dim=-1, eps=1e-5)
        context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5) 

        mem_g = torch.sigmoid(context_g) * (1.0 / seq_len)
        # 预测的 V
        v_retrieved: torch.Tensor = context_k @ self.mem

        # 计算真实 V 与预测 V 的 Delta
        delta_v = context_v - v_retrieved
        
        v_dyn = mem_g * delta_v
        
        # 外积更新
        delta_mem: torch.Tensor = torch.bmm(context_k.transpose(-1, -2), v_dyn)
        
        # 记忆更新
        next_mem: torch.Tensor = self.mem + delta_mem

        if not lock_mem:
            self._pending_mem = next_mem.mean(0, keepdim=True)
        
        # 历史记忆 
        mem_out_prev = context_q @ self.mem
        
        # QK 的标准注意力打分矩阵
        scores = torch.bmm(context_q, context_k.transpose(-1, -2)) 
        
        # 动态生成 Causal Mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril_()
        scores.masked_fill_(~mask, 0.0)
        
        mem_out_delta: torch.Tensor = torch.bmm(scores, v_dyn)
        
        # 合并输出
        mem_out = mem_out_prev
        mem_out = mem_out + mem_out_delta

        # 动态门控
        mem_out = mem_out * self.act(context_d)
        mem_out = self.hidden_norm(mem_out)
        mem_out = self.act(self.w_o(mem_out))
    
        lora_output = mem_out @ self.lora_B.T
        output = base_output + lora_output
   
        if len(orig_shape) == 2:
            output = output.squeeze(1)
        elif len(orig_shape) > 3:
            out_shape = orig_shape[:-1] + (output.shape[-1],)
            output = output.view(out_shape)
        
        return output


class Attention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim
        self.dropout = dropout
        
        self.qkv_proj = OuroCell(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.gate = OuroCell(embed_dim, embed_dim, bias=False)
        self.out_proj = OuroCell(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv: torch.Tensor = self.qkv_proj(x, lock_mem=lock_mem)
        
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k  = self.q_norm(q), self.k_norm(k)
        
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        gate = torch.sigmoid(self.gate(x, lock_mem=lock_mem))
        output = self.out_proj(gate * attn_output, lock_mem=lock_mem)
        
        return output


class GateAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):  
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim

        self.dropout = dropout
        
        self.q_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.kv_proj = OuroCell(embed_dim, 2 * embed_dim, bias=False)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q: torch.Tensor = self.q_proj(x, lock_mem=lock_mem)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if state.dim() == 2:
            state = state.unsqueeze(1) 
            
        kv: torch.Tensor = self.kv_proj(state, lock_mem=lock_mem)
        kv = kv.view(batch_size, 1, 2, self.num_heads, self.head_dim)
    
        kv = kv.permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        q, k = self.q_norm(q), self.k_norm(k)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
    
        gate = torch.sigmoid(scores)
        
        if self.training and self.dropout > 0.0:
            gate = F.dropout(gate, p=self.dropout)
            
        attn_output = gate * v  
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return attn_output


class OuroStateAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim

        self.dropout = dropout

        self.state_proj = OuroCell(self.embed_dim, self.embed_dim)

        self.gate_attn = GateAttention(self.embed_dim, self.dropout)
        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(self.embed_dim, self.dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        state = state.unsqueeze(1)
        state = self.state_proj(state, lock_mem=lock_mem)

        state_injection: torch.Tensor = self.gate_attn(self.in_norm(x), state, lock_mem=lock_mem) 
        x = x + state_injection
        return state_injection + self.attn(self.norm(x), lock_mem=lock_mem)


class OuroDepthAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = embed_dim // self.head_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.q_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.kv_proj = OuroCell(embed_dim, 2 * embed_dim, bias=False)
        self.o_proj = OuroCell(embed_dim, embed_dim, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.o_proj.linear.weight)

    def forward(self, active_c: torch.Tensor, history_states: list[torch.Tensor], lock_mem: bool = False) -> torch.Tensor:
        batch_size = active_c.shape[0]
        seq_len = history_states[0].shape[1]
        num_layers = len(history_states)

        H = torch.stack(history_states, dim=2)
        
        q_norm = self.norm_q(active_c) 
        q: torch.Tensor = self.q_proj(q_norm, lock_mem=lock_mem)
        
        q = q.view(batch_size, 1, self.num_heads, 1, self.head_dim)

        H_norm = self.norm_kv(H)
        kv: torch.Tensor = self.kv_proj(H_norm, lock_mem=lock_mem)
        
        kv = kv.view(batch_size, seq_len, num_layers, 2, self.num_heads, self.head_dim)
        kv = kv.permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.squeeze(-2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.o_proj(out, lock_mem=lock_mem)
    

class OuroTemporalAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = embed_dim // self.head_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.q_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.kv_proj = OuroCell(embed_dim, 2 * embed_dim, bias=False)
        self.o_proj = OuroCell(embed_dim, embed_dim, bias=False)
        
        with torch.no_grad():
            nn.init.zeros_(self.o_proj.linear.weight)

    def forward(self, current_c: torch.Tensor, state_queue: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        batch_size = current_c.shape[0]
        
        q_norm = self.norm_q(current_c)

        q: torch.Tensor = self.q_proj(q_norm, lock_mem=lock_mem)
        q = q.view(batch_size, self.num_heads, 1, self.head_dim)
        
        kv_norm = self.norm_kv(state_queue)
        kv: torch.Tensor = self.kv_proj(kv_norm, lock_mem=lock_mem)
        kv = kv.view(batch_size, state_queue.shape[1], 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.squeeze(2).contiguous().view(batch_size, self.embed_dim)
        return self.o_proj(out, lock_mem=lock_mem)
    

class FFN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim

        self.multiple_of = 256
        hidden_dim = int(2 * (4 * self.embed_dim) / 3)
        self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
        
        self.w12 = OuroCell(self.embed_dim, 2 * self.hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.w3 = OuroCell(self.hidden_dim, self.embed_dim, bias=False)

    def forward(self, x: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        combined_projected = self.w12(x, lock_mem=lock_mem)
        x_w1, x_v = torch.chunk(combined_projected, chunks=2, dim=-1)
        
        swiglu_out = self.act(x_w1) * x_v
        return self.w3(swiglu_out, lock_mem=lock_mem)
    

class OuroSTM(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch

        self.state_attn = OuroStateAttention(self.embed_dim)
        self.act = nn.SiLU()

        self.w = OuroCell(self.embed_dim, self.embed_dim * 4)

        self.c_state: torch.Tensor
        self._pending_c_state: torch.Tensor
        self._runtime_c_state: torch.Tensor | None = None

        self.register_buffer("c_state", torch.zeros(self.max_batch, self.embed_dim))
        self.state_proj = OuroCell(self.embed_dim, self.embed_dim)
        self.ouro_norm = OuroNorm(self.embed_dim)

        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = OuroCell(self.embed_dim, self.embed_dim)

    # def __call__(self, x: torch.Tensor, lock_mem: bool = False, need_print: bool = False):
    #     return self.forward(x, lock_mem, need_print)

    def mem_detach(self):
        self._runtime_c_state = self._runtime_c_state.detach()
            
        with torch.no_grad():
            batch_size = self._runtime_c_state.shape[0]
            self.c_state[:batch_size].copy_(self._runtime_c_state)

    def mem_clear(self):
        self._runtime_c_state = None
        self.c_state.zero_()

    def mem_sync(self):
        self._runtime_c_state = self._pending_c_state    
        del self._pending_c_state

    def active_c(self, batch_size: int):
        if self._runtime_c_state is None:
            active_c = self.c_state[:batch_size].detach().clone()
        else:
            active_c = self._runtime_c_state
        return active_c

    def forward(self, x: torch.Tensor, lock_mem: bool = False, need_print: bool = False):
        batch_size, seq_len, _ = x.shape

        active_c = self.active_c(batch_size)

        if need_print:
            with torch.no_grad():
                print('输入x: ', x.norm(), ' 状态范数: ', active_c.norm())

        state_attn = self.state_attn(x, active_c, lock_mem=lock_mem)
        x = x + state_attn

        gates = self.w(x, lock_mem=lock_mem) 
        i, f, g, o = torch.chunk(gates, 4, dim=-1)

        i = torch.sigmoid(i)
        g: torch.Tensor = self.act(g)
        o = torch.sigmoid(o)

        v = i * g  
        f_gate = torch.sigmoid(f) 

        c_states = []
        curr_c = active_c 

        # 线性时序扫描
        for t in range(seq_len):
            curr_c = f_gate[:, t, :] * curr_c + v[:, t, :]
            c_states.append(curr_c)

        c = torch.stack(c_states, dim=1)
        h: torch.Tensor = o * c

        # 更新 Buffer
        if not lock_mem:
            c_last = c[:, -1, :] 
            c_last = self.state_proj(c_last, lock_mem=lock_mem)
            self._pending_c_state = self.ouro_norm(c_last) 

        return self.out_proj(self.out_norm(h), lock_mem=lock_mem)


class OuroLayer(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int, need_mem: bool = False, need_stm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // HEAD_DIM

        self.max_batch = max_batch
     
        self.need_mem = need_mem
        self.need_stm = need_stm

        self.act = nn.SiLU()
        self.state_attn = OuroStateAttention(self.embed_dim)

        if self.need_stm:
            self.ouro_stm = OuroSTM(self.embed_dim, self.max_batch)

        # 开启标准的 Delte Rule 实现
        if self.need_mem:
            self._pending_mem = None

            # 全局记忆矩阵 
            self.mem: torch.Tensor
            self.register_buffer('mem', torch.eye(self.embed_dim).unsqueeze(0))

            self._causal_mask: torch.Tensor
            self.register_buffer('_causal_mask', torch.ones(self.embed_dim, self.embed_dim, dtype=torch.bool).tril_(), persistent=False)

            self.mem_norm = nn.LayerNorm(embed_dim)

            self.w_qkvgd = OuroCell(embed_dim, embed_dim * 5)

            self.out_norm = nn.LayerNorm(embed_dim)
            self.w_o = OuroCell(self.embed_dim, self.embed_dim, bias=False)
            self.o = OuroCell(self.embed_dim, self.embed_dim, bias=False)
        
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.w_qkvgd.linear.weight[3 * embed_dim : 4 * embed_dim, :], 
                    mean=0.0, std=0.02
                )
                torch.nn.init.constant_(
                    self.w_qkvgd.linear.bias[3 * embed_dim : 4 * embed_dim], 
                    -6.0
                )

    # def __call__(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False):
    #     return self.forward(x, last_state, lock_mem)
    
    def mem_detach(self):
        if self.need_stm:
            self.ouro_stm.mem_detach()
        if self.need_mem:
            self.mem = self.mem.detach()

    def mem_clear(self):
        if self.need_stm:
            self.ouro_stm.mem_clear()
        if self.need_mem:
            self.mem = torch.zeros(1, self.embed_dim, self.embed_dim)

    def mem_sync(self):
        if self.need_stm:
            self.ouro_stm.mem_sync()

    def forward(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len, _ = x.shape
    
        x = x + self.state_attn(x, last_state, lock_mem=lock_mem)

        if self.need_mem:
            mem_context: torch.Tensor = self.mem_norm(x)
            qkvgd: torch.Tensor= self.w_qkvgd(mem_context, lock_mem=lock_mem)
            context_q, context_k, context_v, context_g, context_d = qkvgd.chunk(5, dim=-1)

            context_q = F.normalize(context_q, p=2, dim=-1, eps=1e-5)
            context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5) 

            mem_g = torch.sigmoid(context_g) * (1.0 / seq_len)

            # 预测的 V
            v_retrieved: torch.Tensor = context_k @ self.mem

            # 计算真实 V 与预测 V 的 Delta
            delta_v = context_v - v_retrieved

            v_dyn = mem_g * delta_v
            
            # 外积更新
            delta_mem: torch.Tensor = torch.bmm(context_k.transpose(-1, -2), v_dyn)
            
            # 记忆更新
            next_mem: torch.Tensor = self.mem + delta_mem

            if not lock_mem:
                self._pending_mem = next_mem.mean(0, keepdim=True)
            
            # 历史记忆 
            mem_out_prev = context_q @ self.mem
            
            # QK 的标准注意力打分矩阵
            scores = torch.bmm(context_q, context_k.transpose(-1, -2)) 
          
            # 标准注意力
            mask = self._causal_mask[:seq_len, :seq_len]
            scores.masked_fill_(~mask, 0.0)
            mem_out_delta: torch.Tensor = torch.bmm(scores, v_dyn)
            
            # 合并输出
            mem_out = mem_out_prev
            mem_out += mem_out_delta

            # 动态门控
            mem_out = mem_out * self.act(context_d)
            mem_out = self.w_o(self.out_norm(mem_out), lock_mem=lock_mem)
            mem_out = self.o(self.act(mem_out), lock_mem=lock_mem)
            x = x + mem_out

            return x, scores, x

        if self.need_stm:
            stm = self.ouro_stm(x, lock_mem=lock_mem)
            return x + stm, None, x + stm

    
class OuroBlock(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int, block_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch
        self.block_layers = block_layers

        self.act = nn.SiLU()

        self.ouro_self_attn_proj = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.ouro_self_attn_norm = nn.LayerNorm(self.embed_dim)

        self.w_v = OuroCell(embed_dim, embed_dim, bias=False)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.ouro_self_attn_output_proj = OuroCell(embed_dim, embed_dim, bias=False)
        self.ouro_self_attn_gate = OuroCell(embed_dim, embed_dim, bias=False)

        self.ouro_layers: nn.ModuleList[OuroLayer] = nn.ModuleList([
            OuroLayer(self.embed_dim, self.max_batch, (_!=0), _==0) for _ in range(self.block_layers)
        ])

        self.ffn = FFN(self.embed_dim)

        self.norm = nn.LayerNorm(self.embed_dim)

    # def __call__(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False):
    #     return self.forward(x, last_state, lock_mem)

    def mem_detach(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_detach()

    def mem_clear(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_clear()

    def mem_sync(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_sync()
    
    def forward(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len, _ = x.shape
        residual = x

        ouro_self_attn = torch.tensor(0.0)

        out_list = []
        for layer in self.ouro_layers:
            layer: OuroLayer
            if layer.need_mem:
                x, attn, out = layer(x, last_state, lock_mem)
                ouro_self_attn = ouro_self_attn + attn
            if layer.need_stm:
                x, _, out = layer(x, last_state, lock_mem)

            out_list.append(out)
            inner_residual = x

        # 涌现注意力 (Emergent Attention)
        scale_factor: torch.Tensor = self.embed_dim**(-0.5)

        ouro_self_attn_residual = ouro_self_attn

        x_proj: torch.Tensor = torch.matmul(x, self.ouro_self_attn_proj) * scale_factor
        ouro_self_attn = torch.bmm(ouro_self_attn_residual, x_proj)
        
        ouro_self_attn: torch.Tensor = self.act(ouro_self_attn)
        ouro_self_attn_normed: torch.Tensor = self.ouro_self_attn_norm(ouro_self_attn)

        v_states: torch.Tensor = self.w_v(self.v_norm(residual), lock_mem=lock_mem)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        attn_bias: torch.Tensor = (ouro_self_attn_residual * scale_factor).masked_fill(~causal_mask, float('-inf'))

        ouro_self_attn_output = F.scaled_dot_product_attention(
            ouro_self_attn_normed.unsqueeze(1),
            ouro_self_attn_normed.unsqueeze(1),
            v_states.unsqueeze(1),
            attn_mask=attn_bias.unsqueeze(1),
            scale=scale_factor
        ).squeeze(1) 
           
        gate = torch.sigmoid(self.ouro_self_attn_gate(residual, lock_mem=lock_mem))
        ouro_self_attn_output: torch.Tensor = self.ouro_self_attn_output_proj(gate * ouro_self_attn_output, lock_mem=lock_mem)

        x = inner_residual + ouro_self_attn_output

        # 标准输出
        x = x + self.ffn(self.norm(x), lock_mem=lock_mem)
        out_list.append(x)

        return x, out_list
    

class Ouro(nn.Module):
    """
    Ouro 标准模型
    """
    def __init__(self, embed_dim: int, max_batch: int, blocks: int, block_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch
        self.blocks = blocks

        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.in_attn = Attention(self.embed_dim)
        self.in_ffn_norm = nn.LayerNorm(self.embed_dim)
        self.in_ffn = FFN(self.embed_dim)

        self.temporal_queue_len = 65
        self.c_state_queue: torch.Tensor
        self.register_buffer("c_state_queue", torch.zeros(max_batch, self.temporal_queue_len, self.embed_dim))
        self.temporal_attn = OuroTemporalAttention(self.embed_dim)
        self.c_state_norm = nn.LayerNorm(self.embed_dim)
        self.active_c_norm = nn.LayerNorm(self.embed_dim)
        self.state_ffn = FFN(self.embed_dim)
    
        self.ouro_blocks = nn.ModuleList([
            OuroBlock(self.embed_dim, self.max_batch, block_layers) for _ in range(blocks)
        ])

        self.attnres_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(self.embed_dim)) for _ in range(self.blocks - 1)
        ])
        self.attnres_k_norm = nn.LayerNorm(self.embed_dim)
        self.final_depth_attn = OuroDepthAttention(self.embed_dim)

        self.m_ffn_norm = nn.LayerNorm(self.embed_dim)
        self.m_ffn = FFN(self.embed_dim)

        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.out_attn = Attention(self.embed_dim)
        self.out_ffn_norm = nn.LayerNorm(self.embed_dim)
        self.out_ffn = FFN(self.embed_dim)

        self.stm = OuroSTM(self.embed_dim, self.max_batch)

    # def __call__(self, x: torch.Tensor, lock_mem: bool = False):
    #     return self.forward(x, lock_mem)

    def mem_detach(self):
        self.stm.mem_detach()
        self.c_state_queue = self.c_state_queue.detach()
        
        # 兼容处理所有新引入的 OuroCell
        for module in self.modules():
            if isinstance(module, OuroCell):
                module.mem_detach()
                
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_detach()

    def mem_clear(self):
        self.stm.mem_clear()
        self.c_state_queue.zero_()
        
        # 兼容处理所有新引入的 OuroCell
        for module in self.modules():
            if isinstance(module, OuroCell):
                module.mem_clear()
                
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_clear()

    def mem_sync(self):
        self.stm.mem_sync()

        for block in self.ouro_blocks:
            block: OuroBlock
            block.mem_sync()

        pending_mems = []
        mem_layers = []

        # 统一收集包含记忆机制的 OuroCell 和原有 OuroLayer 的 pending_mem
        for module in self.modules():
            if isinstance(module, (OuroCell, OuroLayer)):
                if hasattr(module, '_pending_mem') and module._pending_mem is not None:
                    pending_mems.append(module._pending_mem)
                    mem_layers.append(module)

        if len(pending_mems) == 0:
            return

        if dist.is_initialized():
            with torch.no_grad():
                # 1. 记录各自的 shape 和 元素总数 (numel)
                shapes = [m.shape for m in pending_mems]
                sizes = [m.numel() for m in pending_mems]
                
                # 2. 将所有的 tensor 展平并拼接成一个 1D Tensor，实现一次性高效全量通信
                flat_mems = torch.cat([m.flatten() for m in pending_mems])
                
                # 3. 进行分布式的 all_reduce 操作
                dist.all_reduce(flat_mems, op=dist.ReduceOp.SUM)
                flat_mems = flat_mems / dist.get_world_size()
                
                # 4. 根据之前的 sizes 拆分，并还原回它们原本的形状
                synced_mems_split = torch.split(flat_mems, sizes)
                synced_mems = [m.view(s) for m, s in zip(synced_mems_split, shapes)]

            for i, layer in enumerate(mem_layers):
                local_mem: torch.Tensor = pending_mems[i]
                synced_mem = synced_mems[i]
                layer.mem = local_mem + (synced_mem - local_mem).detach()
                layer._pending_mem = None
        else:
            for layer in mem_layers:
                layer.mem = layer._pending_mem
                layer._pending_mem = None

    def forward(self, x: torch.Tensor, lock_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = x.shape

        x0 = x

        # 标准输入
        x = x + self.in_attn(self.in_norm(x), lock_mem=lock_mem)
        x = x + self.in_ffn(self.in_ffn_norm(x), lock_mem=lock_mem)

        # 状态获取
        base_active_c = self.stm.active_c(batch_size).to(torch.bfloat16)
       
        queue = self.c_state_queue[:batch_size]
        queue_history = queue[:, :-1, :]

        temporal_context = self.temporal_attn(base_active_c, queue_history, lock_mem=lock_mem)
        active_c = base_active_c + temporal_context
        active_c = active_c + self.state_ffn(self.c_state_norm(active_c), lock_mem=lock_mem)
        active_c_norm = self.active_c_norm(active_c)

        # 计算核心
        history_states = [x0, x]

        for i, block in enumerate(self.ouro_blocks):
            block: OuroBlock
            if i > 0:
                stacked_history = torch.stack(history_states, dim=2)
                
                keys = self.attnres_k_norm(stacked_history) 
                values = stacked_history 
                q = self.attnres_queries[i - 1] 

                scores = torch.matmul(keys, q) / (self.embed_dim ** 0.5)
                alpha = F.softmax(scores, dim=-1)
                x = torch.sum(alpha.unsqueeze(-1) * values, dim=2)
                                                  
            x, out_list = block(x, active_c_norm, lock_mem)
            history_states.extend(out_list)
            residual = x

        depth_attn_out = self.final_depth_attn(active_c_norm, history_states, lock_mem=lock_mem)

        x = residual + depth_attn_out
        x = x + self.m_ffn(self.m_ffn_norm(x), lock_mem=lock_mem)

        # 标准输出
        x = x + self.out_attn(self.out_norm(x), lock_mem=lock_mem)
        x = x + self.out_ffn(self.out_ffn_norm(x), lock_mem=lock_mem)

        out = self.stm(x, lock_mem=lock_mem, need_print=False)

        # with torch.no_grad():
        #     print('最终输出范数: ', out.norm())

        # 状态更新
        if not lock_mem:
            new_c = self.stm._pending_c_state 
            next_queue = torch.roll(self.c_state_queue.clone(), shifts=-1, dims=1)
            next_queue[:batch_size, -1, :] = new_c
            self.c_state_queue = next_queue

        return out
