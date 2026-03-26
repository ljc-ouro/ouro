import os
import json
import torch

from ouro_core import ByteTokenizer
    

class StreamLoader:
    """
    流式数据加载器, 多端点并行流读取
    """
    def __init__(self, patch_size: int, chunk_size: int, datasets: str):
        self.datasets = datasets
        self.chunk_size = chunk_size
        self.patch_size = patch_size + 1
        self.tokenizer = ByteTokenizer()
        
        print(f"📦 初始化流式数据加载: {self.datasets}...")
        
        # 为每个 chunk 维护 buffer 和迭代器
        self.buffers = [[] for _ in range(self.chunk_size)]
        self.iterators = []
        
        # 计算文件等分点
        file_size = os.path.getsize(self.datasets)
        step_size = file_size // self.chunk_size
        
        for i in range(self.chunk_size):
            start_offset = i * step_size
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
                        example = json.loads(decoded_line)
                        text = example.get('text', '')
                        if text:
                            text = text.replace('<|im_end|>', '')
                            yield from self.tokenizer.encode(text)
                            yield self.tokenizer.eos_token_id
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
            
            # 末尾从头循环
            start_offset = 0

    def get_batch(self) -> torch.Tensor:
        # 从 n 个流中分别读取 patch_size 个 token
        for i in range(self.chunk_size):
            while len(self.buffers[i]) < self.patch_size:
                self.buffers[i].append(next(self.iterators[i]))
        
        batch_bytes = []
        for i in range(self.chunk_size):
            batch_bytes.append(self.buffers[i][:self.patch_size])
            self.buffers[i] = self.buffers[i][self.patch_size:]

        # [chunk_size, patch_size]
        input_patches = torch.tensor(batch_bytes, dtype=torch.long)
        return input_patches