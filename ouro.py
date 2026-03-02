import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
from torch.utils.checkpoint import checkpoint
from data_tools import Config, StateTransformerConfig, SQRT2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, timescale: float = 500000.0):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(timescale) / d_model))
        self.div_term: torch.Tensor
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 1:
            indices = indices.unsqueeze(0).expand(x.size(0), -1)
            
        pos_expanded = indices.unsqueeze(-1).float()
        
        # 计算相位
        phase = pos_expanded * self.div_term
        
        # 构造 PE 向量并注入
        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)
        
        return x + pe


class NeuralOscillator(nn.Module):
    """
    神经节律器
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        oscillator_config = self.config.oscillator_config

        self.noise = oscillator_config.noise_std
        self.amp = oscillator_config.amplitude

    def forward(self, prev_state: torch.Tensor, current_cycle_len: int):
        # fp32
        with torch.amp.autocast('cuda', enabled=False):
            prev_state_fp32 = prev_state.float()
            
            # 计算当前步进的角速度
            omega = 2 * math.pi / current_cycle_len
            
            cos_w = math.cos(omega)
            sin_w = math.sin(omega)
            
            # 动态构建矩阵 [2, 2]
            rot_matrix = torch.tensor([
                [cos_w, -sin_w],
                [sin_w,  cos_w]
            ], device=prev_state.device, dtype=torch.float32)
            
            # 旋转
            state_rot = prev_state_fp32 @ rot_matrix
            
            # 噪声
            noise = torch.randn_like(state_rot) * self.noise
            state_rot = state_rot + noise
            
            # 归一化
            norm = torch.norm(state_rot, dim=1, keepdim=True) + 1e-6
            next_state_fp32 = state_rot / norm

        return next_state_fp32.to(prev_state.dtype)

    def get_init_state(self):
        # 初始化
        theta = torch.rand(1, device=self.config.device, dtype=torch.float32) * 2 * math.pi
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y], dim=1)
    

class Hippocampus(nn.Module):
    """
    海马体: 用 Transformer 更新 [Mem, State] 
    """
    def __init__(self, config: Config):
        super().__init__()
        hippo_cfg = config.hippo_config
        self.mem_len = config.brain_config.mem_len
        self.states_len = config.brain_config.states_len
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hippo_cfg.embed_dim,
                nhead=hippo_cfg.heads,
                dim_feedforward=hippo_cfg.dff,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(hippo_cfg.layers)
        ])
        
        self.pos_encoder = PositionalEncoding(hippo_cfg.embed_dim)
        
        self.input_norm = nn.LayerNorm(hippo_cfg.embed_dim)
        self.output_norm = nn.LayerNorm(hippo_cfg.embed_dim)

    def forward(self, mem: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = mem.device
        
        combined = torch.cat([mem, state], dim=1)
        combined = self.input_norm(combined)
        
        idx_mem = torch.arange(-self.mem_len, 0, dtype=torch.float, device=device)
        idx_state = torch.arange(0, self.states_len, dtype=torch.float, device=device)
        
        all_indices = torch.cat([idx_mem, idx_state])
        
        # 位置编码
        hidden = self.pos_encoder(combined, all_indices)
        
        for layer in self.layers:
            hidden = checkpoint(layer, hidden, use_reentrant=False)
            
        out = self.output_norm(hidden)
        
        new_mem = out[:, :self.mem_len, :]
        new_state = out[:, self.mem_len:, :]
        
        # 归一化
        new_mem = F.normalize(new_mem, p=2, dim=-1, eps=1e-5)
        new_state = F.normalize(new_state, p=2, dim=-1, eps=1e-5)
        
        return new_mem, new_state
    

class Hypothalamus(nn.Module):
    """
    下丘脑: 根据状态分泌激素实现状态对整个网络的控制
    """
    def __init__(self, config: Config):
        super().__init__()
        # 维度: 状态/梦境内容的维度 (D) + 节律器维度 (2)
        total_input_dim = config.embed_dim + 2
        self.hormone_dim = config.embed_dim // 4

        self.context_norm = nn.LayerNorm(config.embed_dim)
        
        # 激素网络
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, total_input_dim),
            nn.GELU(), 
            nn.Linear(total_input_dim, self.hormone_dim * 4),
            nn.GELU(), 
            nn.Linear(self.hormone_dim * 4, self.hormone_dim * 2),
            nn.GELU(), 
            nn.Linear(self.hormone_dim * 2, self.hormone_dim),
            nn.GELU(),
            nn.LayerNorm(self.hormone_dim),
            nn.Tanh() 
        )

        self.buffer_norm = nn.LayerNorm(self.hormone_dim)
        
        # 激素缓存
        self.current_hormone: torch.Tensor
        self.register_buffer("current_hormone", torch.zeros(1, 1, self.hormone_dim))

    def get_hormone(self, state_input: torch.Tensor, osc_state: torch.Tensor) -> torch.Tensor:
        # 状态压缩
        if state_input.dim() == 3:
            context = state_input.mean(dim=1) 
        else:
            context = state_input

        # 归一化
        context = self.context_norm(context)
        
        # 节律
        combined_input = torch.cat([context, osc_state], dim=1) 
        
        # 计算激素调节量
        _hormone_delta: torch.Tensor = self.network(combined_input)
        hormone_delta = _hormone_delta.unsqueeze(1)

        hormone_delta = F.normalize(input=hormone_delta, p=2, dim=-1, eps=1e-5)
        
        # 当前的激素水平 
        new_hormone = self.buffer_norm(0.9 * self.current_hormone + 0.1 * hormone_delta)
        
        # 更新内部缓存
        self.current_hormone = F.normalize(input=new_hormone, p=2, dim=-1, eps=1e-5)
            
        return self.current_hormone
    

class HormoneReceptor(nn.Module):
    """
    激素受体: 通用层包装器, 使得输出受到激素调节
    """
    def __init__(self, backbone: nn.Module, config: StateTransformerConfig):
        super().__init__()
        self.backbone = backbone
    
        self.dim = config.embed_dim
        self.hormone_dim = self.dim // 4

        # 动态推断输入输出维度
        if isinstance(backbone, nn.Linear):
            self.in_dim = backbone.in_features
            self.out_dim = backbone.out_features
        else:
            self.in_dim = config.embed_dim
            self.out_dim = config.embed_dim
            
        # Query 生成器
        self.compress_factor = self.in_dim // self.dim
        self.query_norm = nn.LayerNorm(self.dim)

        # Output 投影器 
        self.expand_factor = self.out_dim // self.dim
        self.proj_gate = nn.Parameter(torch.randn(self.expand_factor, self.dim))
        self.proj_norm = nn.LayerNorm(self.out_dim)

        # 激素转化网络
        self.hormone_network = nn.Sequential(
            nn.Linear(self.hormone_dim, self.hormone_dim),
            nn.SiLU(), 
            nn.Linear(self.hormone_dim, self.dim),
            nn.LayerNorm(self.dim)
        )

    def _generate_query(self, x: torch.Tensor, hormone: torch.Tensor) -> torch.Tensor:
        """
        从输入 x 生成查询向量
        """
        B, S, _ = x.shape
        x_folded = x.view(B, S, self.compress_factor, self.dim)
        
        # 加权求和 
        query = x_folded.sum(dim=2) 

        hormone = self.hormone_network(hormone)
            
        return self.query_norm(query + hormone)

    def _project_output(self, query: torch.Tensor) -> torch.Tensor:
        """
        将 query 投影回输出空间
        """
        B, S, _ = query.shape
        out_expanded = query.unsqueeze(2).expand(B, S, self.expand_factor, self.dim) * self.proj_gate
       
        injection = out_expanded.reshape(B, S, -1) 
        active = torch.tanh(self.proj_norm(injection))
            
        return active

    def forward(self, x: torch.Tensor, hormone: torch.Tensor) -> torch.Tensor:
        # 主干计算
        y_logic = self.backbone(x) 
        
        # 生成 Query
        query = self._generate_query(x, hormone)
        
        # 投影回输出 
        active = self._project_output(query)
        y = y_logic * active
      
        return y


class HormoneTransformerLayer(nn.Module):
    """
    支持激素注入的 Transformer Layer
    """
    def __init__(self, config: StateTransformerConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.use_cross_attn = config.use_cross_attn 
        
        # Attention
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(config.embed_dim, config.heads, batch_first=True, dropout=0.1)

        # 交叉注意力
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(config.embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                    embed_dim=config.embed_dim, 
                    num_heads=config.heads, 
                    kdim=config.embed_dim,  
                    vdim=config.embed_dim,  
                    batch_first=True, 
                    dropout=0.1
                )
        
        # FFN & Norm 使用 HormoneReceptor 包裹
        self.norm2 = HormoneReceptor(nn.LayerNorm(config.embed_dim), config)
        
        self.linear1 = HormoneReceptor(nn.Linear(config.embed_dim, config.dff), config)
        self.linear2 = HormoneReceptor(nn.Linear(config.dff, config.embed_dim), config)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, hormone: torch.Tensor, src_mask=None, context: Optional[torch.Tensor]=None):
        # Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=src_mask, need_weights=False)
        x = residual + self.dropout(x)

        # 交叉注意力
        if self.use_cross_attn and context is not None:
            residual = x
            x = self.norm_cross(x)
            x, _ = self.cross_attn(query=x, key=context, value=context, need_weights=False)
            x = residual + self.dropout(x)
        
        # FFN 
        residual = x
        
        # 注入激素到 Norm2
        x = self.norm2(x, hormone)
        
        # 注入激素到 FFN
        x = self.linear1(x, hormone)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x, hormone)
        
        return residual + x
    

class SelfEncoder(nn.Module):
    """
    显式的自我表征, 视作稳定的人格锚点
    """
    def __init__(self, config: Config):
        super().__init__()
        self.self_dim = config.embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.self_dim, self.self_dim // 2),
            nn.GELU(),
            nn.Linear(self.self_dim // 2, self.self_dim),
            nn.Tanh()
        )

        self.norm = nn.LayerNorm(self.self_dim)

        # 持久化
        self.current_self: torch.Tensor
        self.register_buffer(
            "current_self",
            torch.zeros(1, 1, self.self_dim)
        )

    def forward(self, state: torch.Tensor, need_update_self=False):
        """
        state: [B, S, D]
        """
        if need_update_self:
            # 压缩 state → 全局自我描述
            state_summary = state.mean(dim=1)
            delta_self = self.encoder(state_summary).unsqueeze(1)

            # 更新
            new_self = 0.95 * self.current_self + 0.05 * delta_self
            self.current_self = self.norm(new_self)

        return self.current_self


class StateTransformer(nn.Module):
    """
    结构: Mem * [State_Read | Input | State_Write]
    """
    def __init__(self, config: StateTransformerConfig, self_encoder: SelfEncoder):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        # 自我表征
        self.self_encoder = self_encoder
        
        # 输入归一化 
        self.input_norm = nn.LayerNorm(config.embed_dim)
        self.state_norm = nn.LayerNorm(config.embed_dim) 
        self.mem_norm = nn.LayerNorm(config.embed_dim)
        
        # 位置编码
        self.max_logical_pos = config.states_len * 12
        self.pos_encoder = PositionalEncoding(config.embed_dim)
        
        # 状态槽位 Embedding, 区分同一个 State 在 Read 和 Write 位置的不同角色
        self.read_slot_emb = nn.Parameter(torch.randn(1, config.states_len, config.embed_dim))
        self.write_slot_emb = nn.Parameter(torch.randn(1, config.states_len, config.embed_dim))
        nn.init.normal_(self.read_slot_emb, std=0.02)
        nn.init.normal_(self.write_slot_emb, std=0.02)
        
        self.layers = nn.ModuleList([
            HormoneTransformerLayer(config) 
            for _ in range(config.layers)
        ])

        # 输出归一化
        self.text_output_norm = HormoneReceptor(nn.LayerNorm(config.embed_dim), config)
        self.state_output_norm = HormoneReceptor(nn.LayerNorm(config.embed_dim), config)
        
    def _create_sandwich_mask(self, total_len, input_len, device):
        """
        Prefix (Read): 内部全互联
        Input: Causal (看 Prefix + 之前的 Input)
        Write: Full Context (看前面所有)
        """
        end_read = self.config.states_len
        end_input = end_read + input_len
        
        mask = torch.ones((total_len, total_len), device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        
        mask[:end_read, :end_read] = False
        mask[end_input:, :] = False 
        
        return mask

    def forward(self, x_chunk: torch.Tensor, prev_memory: torch.Tensor, prev_state: torch.Tensor, hormone: torch.Tensor, need_update_self=False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        
        _, inputs_len, _ = x_chunk.shape
        device = x_chunk.device
        
        x_input_norm = self.input_norm(x_chunk)
        mem_norm = self.mem_norm(prev_memory)    # Mem 将作为 context 
        state_norm = self.state_norm(prev_state)
        
        part_read = state_norm + self.read_slot_emb
        part_input = x_input_norm
        
        self_encode = self.self_encoder(state_norm, need_update_self)
        part_write = state_norm + self.write_slot_emb + self_encode.expand(-1, self.config.states_len, -1)
        
        combined_input = torch.cat([part_read, part_input, part_write], dim=1)
        
        # 位置索引
        idx_read = torch.zeros(self.config.states_len, dtype=torch.float, device=device)
        idx_input = torch.arange(1, inputs_len + 1, dtype=torch.float, device=device)
        idx_write = torch.full((self.config.states_len,), self.max_logical_pos, dtype=torch.float, device=device)
        
        all_indices = torch.cat([idx_read, idx_input, idx_write])
        hidden = self.pos_encoder(combined_input, all_indices)
        
        total_len = hidden.size(1)
        mask = self._create_sandwich_mask(total_len, inputs_len, device)
        
        for layer in self.layers:
            hidden = checkpoint(layer, hidden, hormone, mask, mem_norm, use_reentrant=False)
            
        # 偏移量更新 
        start_input = self.config.states_len
        end_input = start_input + inputs_len
        
        text_output_raw = hidden[:, start_input : end_input, :]
        new_state_raw = hidden[:, end_input:, :]
        
        text_output = self.text_output_norm(text_output_raw, hormone)
        text_output = F.normalize(text_output, p=2, dim=-1, eps=1e-5)
        
        new_state = self.state_output_norm(new_state_raw, hormone)
        new_state = F.normalize(new_state, p=2, dim=-1, eps=1e-5)
        
        return text_output, new_state
    

class Compressor(nn.Module):
    """
    压缩器: 将字节流压缩为向量
    """
    def __init__(self, patch_size: int, byte_emb_dim: int, model_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_emb_dim = byte_emb_dim
        
        # 捕捉局部字节组合
        self.resonator = nn.Sequential(
            nn.Conv1d(
                in_channels=byte_emb_dim, 
                out_channels=model_dim * 2, # 升维
                kernel_size=8,              # 感受野
                padding=2,
                groups=1                    # 全通道混合
            ),
            nn.GELU(),
            nn.BatchNorm1d(model_dim * 2)   
        )

        # 压缩回原维度
        self.condenser = nn.Sequential(
            nn.Conv1d(model_dim * 2, model_dim, kernel_size=1),
            nn.GELU()
        )

        # 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, model_dim, patch_size)) 
        nn.init.normal_(self.pos_emb, std=0.02)

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=1), 
            nn.GELU(),
        )

        # Start + Max + End -> model_dim
        self.boundary_fusion = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim * 2), 
            nn.LayerNorm(model_dim * 2),             
            nn.GELU(),                               
            nn.Linear(model_dim * 2, model_dim)      
        )

        nn.init.xavier_uniform_(self.boundary_fusion[-1].weight, gain=0.1)

        # 输出投影
        self.proj_norm = nn.LayerNorm(model_dim)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, S, _ = x_flat.shape
        
        # [B*S, Byte_Emb, Patch_Size]
        x = x_flat.view(B * S, self.patch_size, self.byte_emb_dim).transpose(1, 2)
        
        x = self.resonator(x) 
        x = self.condenser(x)
        L = x.size(2)
        
        feat_start = x[:, :, 0]

        pos = self.pos_emb[:, :, :L]
        x_pos = x + pos
        x_pos = self.fusion_mlp(x_pos)

        feat_body = x_pos.mean(dim=-1)

        feat_end = x[:, :, -1]
        x_combined = torch.cat([feat_start, feat_body, feat_end], dim=-1)

        x = self.boundary_fusion(x_combined)
        
        # 归一化
        x = self.proj_norm(x)
        return x.view(B, S, -1)


class Sensor(nn.Module):
    """
    Sensor: 感受器, 将输入转化成状态扰动
    """
    def __init__(self, config: Config, self_encoder: SelfEncoder):
        super().__init__()
        self.pad_token_id = config.tokenizer.pad_token_id
        self.patch_size = config.patch_size
        self.byte_emb_dim = config.byte_embed_dim
        self.model_dim = config.embed_dim 
        
        self.byte_embedding = nn.Embedding(config.byte_vocab_size, self.byte_emb_dim)

        # 基础压缩组件
        self.compressor = Compressor(
            patch_size=self.patch_size,
            byte_emb_dim=self.byte_emb_dim,
            model_dim=self.model_dim
        )
        
        # 使用共享记忆的 StateTransformer 作为上下文混合器
        self.context_mixer = StateTransformer(config.sensor_config, self_encoder)
        
        # 初始化
        nn.init.normal_(self.byte_embedding.weight, std=0.02)

    def forward(self, byte_patches: torch.Tensor, prev_mem: torch.Tensor, prev_state: torch.Tensor, hormone: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = byte_patches.shape
        
        x_emb: torch.Tensor = self.byte_embedding(byte_patches)
        mask = (byte_patches != self.pad_token_id).float().unsqueeze(-1)
        x_emb = x_emb * mask

        x_flat = x_emb.view(B, S, -1)
        x_latent = self.compressor(x_flat) # 无语义, 纯特征
        x_latent = F.normalize(x_latent, p=2, dim=-1)
        
        # 使用 StateTransformer 注入语义, 你感知到的是由你决定的
        features, _ = self.context_mixer(x_latent, prev_mem, prev_state, hormone)
        
        # 返回增强后的特征流, Sensor 不改变记忆
        return features
    

class Decompressor(nn.Module):
    """
    解压器: 将 Brain 的 1 个向量扩展为 K 个锚点
    """
    def __init__(self, input_dim: int, num_anchors: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_anchors = num_anchors
        
        self.expander = nn.Linear(input_dim, input_dim * num_anchors)
        
        # 初始锚点 
        self.sos_anchor = nn.Parameter(torch.randn(1, 1, num_anchors, input_dim))
        nn.init.normal_(self.sos_anchor, std=0.02)

    def forward(self, thought_stream: torch.Tensor, prev_thought: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = thought_stream.shape
        K = self.num_anchors
        
        current_anchors = self.expander(thought_stream).view(B, S, K, D)
        
        if prev_thought is not None:
            prev_anchors_start = self.expander(prev_thought[:, -1:, :]).view(B, 1, K, D)
        else:
            prev_anchors_start = self.sos_anchor.expand(B, 1, -1, -1)
            
        prev_stream = torch.cat([
            prev_anchors_start,
            current_anchors[:, :-1, :, :]
        ], dim=1)
        
        window_context = torch.cat([prev_stream, current_anchors], dim=2)
        
        context_flat = window_context.view(B * S, 2 * K, D)
        
        return context_flat


class Actor(nn.Module):
    """
    生成字节流 NAR
    """
    def __init__(self, config: Config):
        super().__init__()
        self.patch_size = config.patch_size
        self.vocab_size = config.byte_vocab_size 
        self.mask_id = self.vocab_size           # MASK ID = 258
        self.embed_vocab_size = self.mask_id + 1 # 259
        
        self.dim = config.actor_config.embed_dim
        self.num_anchors = config.actor_config.num_anchors
        
        self.decompressor = Decompressor(config.embed_dim, self.num_anchors)
        self.hormone_proj = nn.Linear(config.embed_dim // 4, self.dim // 4)
        
        self.byte_embedding = nn.Embedding(self.embed_vocab_size, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.patch_size, self.dim))
        
        self.decoder_layer = nn.ModuleList([
            HormoneTransformerLayer(config.actor_config)
            for _ in range(config.actor_config.layers)
        ])

        self.final_norm = HormoneReceptor(nn.LayerNorm(self.dim), config.actor_config)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, thought_stream: torch.Tensor, 
                target_patches: Optional[torch.Tensor] = None, 
                hormone: Optional[torch.Tensor] = None, 
                prev_thought: Optional[torch.Tensor] = None,
                prev_last_tokens: Optional[torch.Tensor] = None,
                force_prefix: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: int = 0,
                min_mask_ratios=0.1
                ) -> torch.Tensor:
        
        B, S, _ = thought_stream.shape
        P = self.patch_size
        device = thought_stream.device

        # 解压 thought 
        context_flat = self.decompressor(thought_stream, prev_thought)
        
        # 展开激素
        if hormone is not None:
            hormone = self.hormone_proj(hormone) 
            H_dim = self.decoder_layer[0].norm2.hormone_dim
            hormone_flat = hormone.expand(-1, S, -1).reshape(B * S, 1, H_dim)
        else:
            hormone_flat = None

        if target_patches is not None:
            clean_targets = target_patches.clone()
            clean_targets[clean_targets == -100] = 0 # 处理 PAD
            
            # 随机 mask
            mask_ratios = torch.rand((B, S, 1), device=device) * (1-min_mask_ratios) + min_mask_ratios 
            mask_ratios[mask_ratios > 0.85] = 1.0
            
            prob_matrix = torch.rand((B, S, P), device=device)
            mask_indices = prob_matrix < mask_ratios
            
            masked_inputs = clean_targets.clone()
            masked_inputs[mask_indices] = self.mask_id
            
            decoder_input = self.byte_embedding(masked_inputs) + self.pos_embedding
            x = decoder_input.view(B * S, P, -1)
            
            for layer in self.decoder_layer:
                x = checkpoint(layer, x, hormone_flat, src_mask=None, context=context_flat, use_reentrant=False)

            out_feat = self.final_norm(x, hormone_flat)
            logits = self.head(out_feat.view(B, S, P, -1))
            
            return logits

        else:
            current_tokens = torch.full((B, S, P), self.mask_id, dtype=torch.long, device=device)

            # 前缀注入
            if force_prefix is not None:
                prefix_len = force_prefix.size(-1)
                current_tokens[:, :, :prefix_len] = force_prefix

            # 第一轮前向
            decoder_input = self.byte_embedding(current_tokens) + self.pos_embedding
            x = decoder_input.view(B * S, P, -1)
            
            for layer in self.decoder_layer:
                x = layer(x, hormone_flat, src_mask=None, context=context_flat)
                
            out_feat = self.final_norm(x, hormone_flat)
            logits_step1 = self.head(out_feat.view(B, S, P, -1)) # [B, S, P, V]

            # 获取第一轮的概率分布
            probs_step1 = F.softmax(logits_step1, dim=-1)
            confidences, predicted_tokens = torch.max(probs_step1, dim=-1) # [B, S, P]

            # 保护强制前缀
            if force_prefix is not None:
                prefix_len = force_prefix.size(-1)
                confidences[:, :, :prefix_len] = float('inf')
                predicted_tokens[:, :, :prefix_len] = force_prefix
            
            mask_ratio = 0.5 
            mask_count = max(1, int(P * mask_ratio))
            
            kth_values = torch.kthvalue(confidences.view(B * S, P), mask_count, dim=-1).values
            thresholds = kth_values.view(B, S, 1)
            
            # 重新生成带 MASK 的混合序列
            re_mask_indices = confidences <= thresholds
            refined_tokens = predicted_tokens.clone()
            refined_tokens[re_mask_indices] = self.mask_id # 低置信度的变回 MASK

            # 第二轮前向传播 
            decoder_input_2 = self.byte_embedding(refined_tokens) + self.pos_embedding
            x2 = decoder_input_2.view(B * S, P, -1)
            
            for layer in self.decoder_layer:
                x2 = layer(x2, hormone_flat, src_mask=None, context=context_flat)
                
            out_feat_2 = self.final_norm(x2, hormone_flat)
            final_logits = self.head(out_feat_2.view(B, S, P, -1))

            if temperature != 1.0:
                final_logits = final_logits / max(temperature, 1e-6)
            
            if top_k > 0:
                v, _ = torch.topk(final_logits, min(top_k, self.vocab_size), dim=-1)
                final_logits[final_logits < v[..., [-1]]] = -float('Inf')
            
            return final_logits

class Brain(nn.Module):
    def __init__(self, config: Config, self_encoder: SelfEncoder):
        super().__init__()
        self.core = StateTransformer(config.brain_config, self_encoder)
        
        # 输出概率
        self.prob_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, config.embed_dim // 4),
            nn.GELU(),
            nn.Linear(config.embed_dim // 4, 1)
        )

    def forward(self, thoughts_stream: torch.Tensor, prev_mem: torch.Tensor, prev_state: torch.Tensor, hormone: torch.Tensor):
        features, new_state = self.core(thoughts_stream, prev_mem, prev_state, hormone, True)
        prob = torch.mean(self.prob_proj(new_state))
        return features, prob, new_state
    

class Ouro(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.oscillator = NeuralOscillator(config)

        self.cycle_counter: torch.Tensor
        self.register_buffer('cycle_counter', torch.tensor(0, dtype=torch.long))

        self.hypothalamus = Hypothalamus(config)
        self.hippocampus = Hippocampus(config)

        self.self_encoder = SelfEncoder(config)
        
        self.sensor = Sensor(config, self.self_encoder)
        self.brain = Brain(config, self.self_encoder)
        self.actor = Actor(config)
        
        self.loss_fct = nn.CrossEntropyLoss()

    def get_init_states(self):
        def _get(cfg: StateTransformerConfig):
            return (torch.zeros(1, cfg.mem_len, cfg.embed_dim, device=self.config.device),
                    torch.zeros(1, cfg.states_len, cfg.embed_dim, device=self.config.device))
        
        m_inn, s_inn = _get(self.config.brain_config)
        t_inn = None
        
        # 节律器状态 
        osc_state = self.oscillator.get_init_state()
        
        return (m_inn, s_inn, t_inn, osc_state)
    
    def detach_internal_states(self):
        """
        在 BPTT 边界手动截断内部的隐式递归状态
        """
        # 截断全局激素
        self.hypothalamus.current_hormone = self.hypothalamus.current_hormone.detach()

        # 截断全局自我
        self.self_encoder.current_self = self.self_encoder.current_self.detach()
    
    def forward(self, input_patches: Optional[torch.Tensor], target_patches: Optional[torch.Tensor]=None, 
                states: Optional[Tuple[torch.Tensor, ...]]=None, is_sleeping_phase: bool = False,
                override_last_tokens: Optional[torch.Tensor] = None,
                force_prefix: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: int = 0,
                min_mask_ratios=0.1
                ) -> tuple[torch.Tensor, Optional[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor], bool]: 
        
        # 状态初始化
        if states is None and input_patches is not None:
             states = self.get_init_states()
        
        brain_mem, brain_state, brain_thought,  osc_state = states
        device = brain_mem.device
        
        # 节律更新
        next_osc_state = self.oscillator(osc_state, self.config.cycle_len)
        self.cycle_counter = self.cycle_counter + 1

        is_sleeping = is_sleeping_phase
        if input_patches is None:
            is_sleeping = True
            
        # 睡眠
        if is_sleeping:
            next_brain_mem, next_brain_state = self.hippocampus(brain_mem, brain_state)
            
            zero_loss = torch.tensor(0.0, device=device, dtype=next_osc_state.dtype)

            return zero_loss, None, (next_brain_mem, next_brain_state, None, next_osc_state), True, zero_loss
        
        # 清醒
        else:
            hormone = self.hypothalamus.get_hormone(brain_state, next_osc_state)
            
            # Sensor
            latent_input = self.sensor(input_patches, brain_mem, brain_state, hormone)
            
            # Brain
            final_thought, prob_loss, next_brain_state = self.brain(latent_input, brain_mem, brain_state, hormone)
            next_brain_mem = brain_mem
            
            # 处理 SOS token
            prev_last_tokens = override_last_tokens if override_last_tokens is not None else input_patches[:, :, -1]

            # Actor
            logits = self.actor(
                final_thought, 
                prev_thought=brain_thought,
                target_patches=target_patches, 
                hormone=hormone, 
                prev_last_tokens=prev_last_tokens,
                force_prefix=force_prefix,
                temperature=temperature,
                top_k=top_k,
                min_mask_ratios=min_mask_ratios
            )
            
            # Loss
            task_loss = torch.tensor(0.0, device=device)
            logits_loss = torch.tensor(0.0, device=device)
            if target_patches is not None:
                flat_logits = logits.view(-1, self.config.byte_vocab_size)
                flat_targets = target_patches.view(-1)
                logits_loss = self.loss_fct(flat_logits, flat_targets)

                task_loss = (1/SQRT2 * prob_loss - SQRT2 * logits_loss) ** 2 + 0.5 * prob_loss ** 2
                
            return task_loss, logits, (next_brain_mem, next_brain_state, final_thought, next_osc_state), False, logits_loss