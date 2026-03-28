from dataclasses import dataclass
from ouro_core import ByteTokenizer
    
    
@dataclass
class Config:
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
    pretrain_chunk_size: int = 41
    pretrain_bptt_size: int = 4
    pretrain_steps: int = 250000

    # SFT 配置
    sft_file: str = f'datasets/sft_mini_512/sft_mini_512_train.jsonl'
    sft_chunk_size: int = 27
    sft_bptt_size: int = 6
    sft_lr: float = 8e-5
    sft_steps: int = 400000


GRIDMAN_MINI = Config()

# GRIDMAN_SMALL = Config(
#     'gridman_small',
#     768,
#     8,
#     2,
#     768,
#     13,
#     4
# )

# GRIDMAN_MEDIUM = Config(
#     'gridman_medium',
#     1024,
#     8,
#     3,
#     1024,
#     4,
#     4
# )

# GRIDMAN_LARGE = Config(
#     'gridman_large',
#     1280,
#     12,
#     3,
#     1280,
#     13,
#     4
# )

# GRIDMAN_XL = Config(
#     'gridman_xl',
#     1600,
#     12,
#     4,
#     1280,
#     13,
#     4
# )

RUNNING_CONFIG = GRIDMAN_MINI