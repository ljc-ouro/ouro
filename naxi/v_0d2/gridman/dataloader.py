import os
import json
import torch

from naxi.v_0d2.gridman.lm_kernel import ByteTokenizer


class StreamLoader:
    def __init__(self, patch_size: int, chunk_size: int, datasets: str, is_sft: bool = False, rank: int = 0, world_size: int = 1):
        self.datasets = datasets
        self.chunk_size = chunk_size
        self.patch_size = patch_size + 1
        self.tokenizer = ByteTokenizer()
        self.is_sft = is_sft
        
        mode = 'SFT' if self.is_sft else 'PRE-TRAIN'
        
        if rank == 0:  # 避免多卡同时打印
            print(f'📦 初始化流式数据加载 [{mode}]: {self.datasets}...')
        
        self.buffers = [[] for _ in range(self.chunk_size)]
        self.iterators = []
        
        file_size = os.path.getsize(self.datasets)

        # 将整个文件划分为 chunk_size * world_size 份
        total_chunks = self.chunk_size * world_size
        step_size = file_size // total_chunks
        
        # 每条流的读取现场, 支持断点恢复:
        #   offset    = 当前正在消费的行的起始字节偏移(始终行对齐)
        #   skip      = 该行已消费的 token 数
        #   skip_line = 初始分区点不在行首, 首次打开需跳过残行
        self.stream_states = []
        for i in range(self.chunk_size):
            # 根据 rank 错开各自的 offset
            start_offset = (rank * self.chunk_size + i) * step_size
            self.stream_states.append({
                'offset': start_offset,
                'skip': 0,
                'skip_line': start_offset > 0,
            })
            self.iterators.append(self._get_stream(i))

    def _get_stream(self, stream_idx: int):
        state = self.stream_states[stream_idx]
        while True: 
            with open(self.datasets, 'rb') as f:
                f.seek(state['offset'])
                if state['skip_line']:
                    f.readline()  # 初始分区点未行对齐, 跳过残行
                    state['skip_line'] = False
                
                while True:
                    line_offset = f.tell()  # 当前行的起始字节偏移
                    line = f.readline()
                    if not line:
                        break  # EOF, 跳出后环绕重读
                    if not line.strip():
                        continue
                    try:
                        example: dict[str, str | list[dict[str, str]]] = json.loads(line.decode('utf-8'))
                        
                        # 整行编码为 (token, mask) 序列, 使行内任意位置都可以精确恢复
                        tokens = []
                        if self.is_sft:
                            for turn in example.get('conversations', []):
                                role = turn['role']
                                content = turn['content']
                                if content == '':
                                    continue
                                if role == 'user':
                                    role_id = self.tokenizer.user_token_id
                                elif role == 'assistant':
                                    role_id = self.tokenizer.assistant_token_id
                                else:
                                    continue 
                                
                                tokens.append((role_id, 0))
                                if role == 'assistant':
                                    # Assistant 的内容, mask=1
                                    for t in self.tokenizer.encode(content):
                                        tokens.append((t, 1))
                                    tokens.append((self.tokenizer.eos_token_id, 1))
                                else:
                                    for t in self.tokenizer.encode(content):
                                        tokens.append((t, 0))
                                    tokens.append((self.tokenizer.eos_token_id, 0))
                        else:
                            # 预训练模式
                            text: str = example.get('text', '').replace('<|im_end|>', '')
                            if text:
                                for t in self.tokenizer.encode(text):
                                    tokens.append((t, 1))
                                tokens.append((self.tokenizer.eos_token_id, 1))
                                
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    
                    if not tokens:
                        continue
                    
                    # 恢复时跳过本行已消费的 token; 每次 yield 前更新现场,
                    # 保证生成器挂起时 state 恰好指向"下一个待产出 token"
                    for idx in range(state['skip'], len(tokens)):
                        state['offset'] = line_offset
                        state['skip'] = idx + 1
                        yield tokens[idx]
                    
                    # 本行消费完毕, 现场推进到下一行行首
                    state['skip'] = 0
                    state['offset'] = f.tell()
            
            # EOF: 回到文件开头循环
            state['offset'] = 0
            state['skip'] = 0
            state['skip_line'] = False

    def state_dict(self) -> dict:
        """导出可落盘的读取现场(断点重训用)"""
        return {
            'buffers': [list(b) for b in self.buffers],
            'stream_states': [dict(s) for s in self.stream_states],
        }

    def load_state_dict(self, state: dict):
        """恢复读取现场并按现场重建迭代器"""
        self.buffers = [list(b) for b in state['buffers']]
        for i, s in enumerate(state['stream_states']):
            self.stream_states[i].update(s)
        self.iterators = [self._get_stream(i) for i in range(self.chunk_size)]

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 分别读取 patch_size (token, mask) 
        for i in range(self.chunk_size):
            while len(self.buffers[i]) < self.patch_size:
                self.buffers[i].append(next(self.iterators[i]))
        
        batch_tokens = []
        batch_masks = []
        for i in range(self.chunk_size):
            chunk_data = self.buffers[i][:self.patch_size]
            batch_tokens.append([x[0] for x in chunk_data])
            batch_masks.append([x[1] for x in chunk_data])
            self.buffers[i] = self.buffers[i][self.patch_size - 1:]

        # Token ID, Mask
        input_patches = torch.tensor(batch_tokens, dtype=torch.long)
        mask_patches = torch.tensor(batch_masks, dtype=torch.long)
        return input_patches, mask_patches