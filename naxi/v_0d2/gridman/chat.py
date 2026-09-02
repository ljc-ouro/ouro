import torch

from naxi.v_0d2.gridman.config import RUNNING_CONFIG, Config
from naxi.v_0d2.gridman.core import Gridman
from naxi.v_0d2.gridman.tools import load_checkpoint


import torch

class GridmanChat:
    def __init__(self, model: Gridman, is_sft: bool = True, config: Config = RUNNING_CONFIG):
        self.model = model
        self.is_sft = is_sft
        self.config = config
        self.tokenizer = config.tokenizer
        self.patch_size = config.patch_size
        self.device = config.device
        self.device_type = config.device_type
        
        self.is_first_turn = True
        self.total_gen_tokens_num = 0
        self.generated_tokens = []
        self.current_patch = []
        # 用于存储最近一次记忆固化（lock_mem=False）产生的最后一个 logit
        self.cached_logits = None

    def _split_valid_utf8(self, patch: list[int]):
        n = len(patch)
        for i in range(1, min(5, n + 1)):
            token = patch[-i]
            if token > 255 or token <= 127: return patch, []
            if 192 <= token <= 247:
                expected = 2 if 192 <= token <= 223 else (3 if 224 <= token <= 239 else 4)
                return (patch, []) if i == expected else (patch[:-i], patch[-i:])
        return patch, []

    @torch.no_grad()
    def chat(self, user_input: str | None, max_len: int = 512, temperature: float = 0.7) -> tuple[str, bool]:
        self.model.eval()
        
        # 处理 Prompt
        if user_input is not None:
            # 构造 prefix
            if self.is_sft:
                prefix = [self.tokenizer.eos_token_id, self.tokenizer.user_token_id] if self.is_first_turn else [self.tokenizer.user_token_id]
                input_ids = prefix + self.tokenizer.encode(user_input) + [self.tokenizer.eos_token_id, self.tokenizer.assistant_token_id]
                self.is_first_turn = False
            else:
                input_ids = ([self.tokenizer.eos_token_id] if self.is_first_turn else []) + self.tokenizer.encode(user_input)
                self.is_first_turn = False

            # 消费输入并处理 Patch 溢出
            for token in input_ids:
                self.current_patch.append(token)
                if len(self.current_patch) == self.patch_size:
                    valid_patch, leftover = self._split_valid_utf8(self.current_patch)
                    if valid_patch:
                        p_tensor = torch.tensor([valid_patch], dtype=torch.long, device=self.device)

                        # 更新记忆
                        with torch.amp.autocast(self.device_type, dtype=torch.bfloat16):
                            logits = self.model(p_tensor, False)
                            self.model.core_ouro.mem_sync()
                        
                        # 缓存最后的 logits
                        self.cached_logits = logits[:, -1, :]
                        self.current_patch = leftover

            self.generated_tokens = []

        # 自回归生成
        for _ in range(max_len):
            # 决定采样来源
            if len(self.current_patch) > 0:
                # 还有上下文，正常前向推理（不固化记忆）
                p_tensor = torch.tensor([self.current_patch], dtype=torch.long, device=self.device)
                with torch.amp.autocast(self.device_type, dtype=torch.bfloat16):
                    logits = self.model(p_tensor, True)
                current_logits = logits[:, -1, :]
            elif self.cached_logits is not None:
                # Patch 刚好填满被清空了，直接用缓存的 Logits
                current_logits = self.cached_logits
                self.cached_logits = None # 用完即弃
            else:
                # 理论上不应到达这里
                break

            # 采样
            if temperature <= 0.0:
                next_token = torch.argmax(current_logits, dim=-1).item()
            else:
                probs = torch.nn.functional.softmax(current_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            self.generated_tokens.append(next_token)
            self.current_patch.append(next_token)
            self.total_gen_tokens_num += 1

            # 检查生成过程中是否填满 Patch
            if len(self.current_patch) == self.patch_size:
                valid_patch, leftover = self._split_valid_utf8(self.current_patch)
                if valid_patch:
                    p_tensor = torch.tensor([valid_patch], dtype=torch.long, device=self.device)
                    with torch.amp.autocast(self.device_type, dtype=torch.bfloat16):
                        logits = self.model(p_tensor, False)
                        self.model.core_ouro.mem_sync()
                    
                    self.cached_logits = logits[:, -1, :]
                    
                    # 解码当前已完成的部分（注意排除 leftover）
                    decode_len = len(self.generated_tokens) - len(leftover)
                    response = self.tokenizer.decode(self.generated_tokens[:decode_len])
                    
                    self.current_patch = leftover
                    self.generated_tokens = list(leftover)
                    
                    # 如果达到了最大长度，强行结束
                    if self.total_gen_tokens_num >= max_len:
                        return response, True
                    return response + '\n', False

            # EOS 检查
            if next_token == self.tokenizer.eos_token_id or self.total_gen_tokens_num >= max_len:
                valid, leftover = self._split_valid_utf8(self.generated_tokens)
                # 如果有无法解码的碎片，直接丢弃，只解码 valid 部分
                response = self.tokenizer.decode(valid if valid else self.generated_tokens)
                self.generated_tokens = []
                self.total_gen_tokens_num = 0
                return response, True


def gridman_chat(is_sft: bool = True):
    config = RUNNING_CONFIG
    device = config.device

    # 加载模型
    grid_man = Gridman(config).to(device)
    load_checkpoint(grid_man, is_sft)

    torch.manual_seed(torch.seed())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch.seed())
    
    # 实例化对话系统
    chat_bot = GridmanChat(grid_man, is_sft, config)
    
    print('\n开启对话 (输入 "quit" 或 "exit" 退出)')
    while True:
        user_input = input('User: ')
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        
        while True:
            response, chat_over = chat_bot.chat(user_input, max_len=4096, temperature=0.7)
            print(response, end='', flush=True) 
            if chat_over:
                print('\n') # 结束换行
                break
            else:
                user_input = None