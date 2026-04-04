import os
import json
import torch

from naxi.v_0d1.gridman.lm_kernel import ByteTokenizer


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
        
        for i in range(self.chunk_size):
            # 根据 rank 错开各自的 offset
            start_offset = (rank * self.chunk_size + i) * step_size
            self.iterators.append(self._get_stream(start_offset))

    def _get_stream(self, start_offset: int):
        while True: 
            with open(self.datasets, 'rb') as f:
                f.seek(start_offset)
                if start_offset > 0:
                    f.readline() 
                
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        decoded_line = line.decode('utf-8')
                        example: dict[str, str | list[dict[str, str]]] = json.loads(decoded_line)
                        
                        if self.is_sft:
                            conversations = example.get('conversations', [])
                            for turn in conversations:
                                role = turn['role']
                                content = turn['content']
                                
                                # 将角色映射为具体的 Token ID
                                if role == 'system':
                                    role_id = self.tokenizer.system_token_id
                                elif role == 'user':
                                    role_id = self.tokenizer.user_token_id
                                elif role == 'assistant':
                                    role_id = self.tokenizer.assistant_token_id
                                else:
                                    continue 
                                
                                if role == 'assistant':
                                    yield (role_id, 0)
                                    
                                    # Assistant 的内容, mask=1
                                    for t in self.tokenizer.encode(content):
                                        yield (t, 1)
                                    yield (self.tokenizer.eos_token_id, 1)
                                else:
                                    yield (role_id, 0)
                                    for t in self.tokenizer.encode(content):
                                        yield (t, 0)
                                    yield (self.tokenizer.eos_token_id, 0)
                                    
                        else:
                            # 预训练模式
                            text: str = example.get('text', '')
                            text = text.replace('<|im_end|>', '')
                            if text:
                                for t in self.tokenizer.encode(text):
                                    yield (t, 1)
                                yield (self.tokenizer.eos_token_id, 1)
                                
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
            
            start_offset = 0

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
            self.buffers[i] = self.buffers[i][self.patch_size:]

        # Token ID, Mask
        input_patches = torch.tensor(batch_tokens, dtype=torch.long)
        mask_patches = torch.tensor(batch_masks, dtype=torch.long)
        return input_patches, mask_patches