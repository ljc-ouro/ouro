from dataclasses import dataclass
import os

import torch
from naxi.v_0d1.gridman.lm_kernel import ByteTokenizer
    
    
@dataclass
class Config:
    # 通用配置
    name: str = 'gridman_mini'

    embed_dim: int = 512
    block_layers: int = 6
    blocks: int = 2

    patch_size: int = 512
    
    # 分词器
    tokenizer: ByteTokenizer = ByteTokenizer()

    # 预训练配置
    pretrain_train_file: str = f'datasets/pretrain/pretrain_train.jsonl'
    pretrain_lr: float = 4e-4
    pretrain_chunk_size: int = 77
    pretrain_bptt_size: int = 2
    pretrain_steps: int = 138000

    # SFT 配置
    sft_file: str = f'datasets/sft_mini_512/sft_mini_512_train.jsonl'
    sft_chunk_size: int = 9
    sft_bptt_size: int = 16
    sft_lr: float = 6.25e-5
    sft_steps: int = 230000

    # 版本号
    version: str = 'v_0d1'

    # 运行信息
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.bfloat16

    # 关键路径
    checkpoint_dir: str = os.path.join(os.getcwd(), 'checkpoints')
    log_dir: str = os.path.join(os.getcwd(), 'log')


GRIDMAN_MINI = Config()

GRIDMAN_SMALL = Config(
    'gridman_small',
    768,
    8,
    2,
    768,
    pretrain_chunk_size=18,
    pretrain_bptt_size=3,
    pretrain_lr=2e-4,
    pretrain_steps=400000
)

GRIDMAN_MEDIUM = Config(
    'gridman_medium',
    1024,
    8,
    3,
    pretrain_chunk_size=15,
    pretrain_bptt_size=2,
    pretrain_steps=350000
)

GRIDMAN_LARGE = Config(
    'gridman_large',
    1280,
    9,
    4,
    1280
)

GRIDMAN_XL = Config(
    'gridman_xl',
    1600,
    12,
    4,
    1280
)

RUNNING_CONFIG = GRIDMAN_MINI