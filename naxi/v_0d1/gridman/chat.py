import torch

from naxi.v_0d1.gridman.config import RUNNING_CONFIG, Config
from naxi.v_0d1.gridman.core import Gridman
from naxi.v_0d1.gridman.tools import load_checkpoint


class GridmanChat:
    def __init__(self, model: Gridman, config: Config = RUNNING_CONFIG):
        """
        初始化对话管理器
        """
        self.model = model
        self.config = config
        self.tokenizer = config.tokenizer
        self.patch_size = config.patch_size
        self.device_type = config.device_type
        self.device = config.device
        
        # 核心对话状态
        self.is_first_turn = True
        self.total_gen_tokens_num = 0
        self.generated_tokens = []
        self.current_patch = []
        
    def _split_valid_utf8(self, patch: list[int]) -> tuple[list[int], list[int]]:
        """
        返回完整的 tokens 列表, 被截断的剩余字节列表
        """
        n = len(patch)
        # UTF-8 字符最多 4 个字节，因此最多往前回溯 4 个 token
        for i in range(1, min(5, n + 1)):
            token = patch[-i]
            
            # 1. 如果是特殊 Token (>255) 或者是单字节 ASCII (<=127)
            if token > 255 or token <= 127:
                return patch, [] # 尾部完全合法，没有被截断
                
            # 2. 如果是 UTF-8 多字节的起始字节 (110xxxxx, 1110xxxx, 11110xxx)
            if 192 <= token <= 247:
                expected_len = 0
                if 192 <= token <= 223: expected_len = 2
                elif 224 <= token <= 239: expected_len = 3
                elif 240 <= token <= 247: expected_len = 4
                
                if i == expected_len:
                    # 刚好包含完整的多字节字符
                    return patch, []
                else:
                    # 字符被截断：前面的部分完整，后面 i 个字节是不完整的序列
                    return patch[:-i], patch[-i:]
                    
        # 3. 如果回溯了 4 个字节全都是延续字节 (128-191)，说明遇到了非标准/错误的字节序列
        # 直接返回原样，交由模型自行容错处理
        return patch, []

    @torch.no_grad()
    def chat(self, user_input: str | None, max_len: int = 512, temperature: float = 0.7) -> tuple[str, bool]:
        """
        SFT 对话函数
        """
        self.model.eval()
        
        if user_input is not None:
            # Prompt Token
            if self.is_first_turn:
                input_ids = (
                    [self.tokenizer.eos_token_id, self.tokenizer.user_token_id] + 
                    self.tokenizer.encode(user_input) + 
                    [self.tokenizer.eos_token_id, self.tokenizer.assistant_token_id]
                )
                self.is_first_turn = False
            else:
                input_ids = (
                    [self.tokenizer.user_token_id] + 
                    self.tokenizer.encode(user_input) + 
                    [self.tokenizer.eos_token_id, self.tokenizer.assistant_token_id]
                )
                
            # 消费用户输入
            the_last = None
            for token in input_ids:
                self.current_patch.append(token)
                
                # 当输入填满一个 patch 窗口时，进行 UTF-8 截断检查并前向更新
                if len(self.current_patch) == self.patch_size:
                    valid_patch, leftover_bytes = self._split_valid_utf8(self.current_patch)
                    
                    if len(valid_patch) > 0:
                        patch_tensor = torch.tensor([valid_patch], dtype=torch.long, device=self.device)
                        # 模型记忆更新, 只用完整的 UTF-8 片段
                        _ = self.model(patch_tensor, lock_mem=False)
                        
                        the_last = [valid_patch[-1]]
                        self.current_patch = leftover_bytes

            if len(self.current_patch) == 0:
                self.current_patch = the_last

        # 自回归生成
        self.generated_tokens = []
        
        for _ in range(self.patch_size):
            patch_tensor = torch.tensor([self.current_patch], dtype=torch.long, device=self.device)
            
            # 窗口未满时仅使用历史记忆 + 窗口上下文进行生成, 不更新记忆
            with torch.amp.autocast(self.device_type, dtype=torch.bfloat16):
                logits = self.model(patch_tensor, lock_mem=True)
                
            next_token_logits = logits[:, -1, :]
            
            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1).item()
            else:
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            self.generated_tokens.append(next_token)
            self.current_patch.append(next_token)
            self.total_gen_tokens_num += 1
                
            # 若在生成过程中填满了 patch
            the_last = None
            if len(self.current_patch) == self.patch_size:
                valid_patch, leftover_bytes = self._split_valid_utf8(self.current_patch)
                
                patch_tensor = torch.tensor([valid_patch], dtype=torch.long, device=self.device)
                with torch.amp.autocast(self.device_type, dtype=torch.bfloat16):
                    _ = self.model(patch_tensor, lock_mem=False)
                
                the_last = [valid_patch[-1]]
                self.current_patch = leftover_bytes

                response = self.tokenizer.decode(self.generated_tokens)
                self.generated_tokens = leftover_bytes

                if len(self.current_patch) == 0:
                    self.current_patch = the_last
                    self.generated_tokens= []

                # 在边界达到最大生成限制
                if self.total_gen_tokens_num >= max_len:
                    self.current_patch.append(self.tokenizer.eos_token_id)
                    self.generated_tokens= []
                    self.total_gen_tokens_num = 0
                    return response, True
                else:
                    return response, False

            # 在 patch 内达到最大生成限制
            if self.total_gen_tokens_num >= max_len:
                valid_patch, leftover_bytes = self._split_valid_utf8(self.generated_tokens)

                self.current_patch.extend(valid_patch)
                self.current_patch.append(self.tokenizer.eos_token_id)

                response = self.tokenizer.decode(valid_patch)

                self.generated_tokens= []
                self.total_gen_tokens_num = 0

                return response, True

            # patch 内终止直接返回
            if next_token == self.tokenizer.eos_token_id:
                response = self.tokenizer.decode(self.generated_tokens)
                self.generated_tokens= []
                self.total_gen_tokens_num = 0
                return response, True
    

def gridman_chat():
    config = RUNNING_CONFIG
    device = config.device

    # 加载模型
    grid_man = Gridman(config).to(device)
    load_checkpoint(grid_man, True)
    
    # 实例化对话系统
    chat_bot = GridmanChat(grid_man, config)
    
    print('\n开启对话 (输入 "quit" 退出)')
    while True:
        user_input = input('User: ')
        if user_input.strip().lower() == 'exit':
            break
        
        while True:
            response, chat_over = chat_bot.chat(user_input, max_len=512, temperature=0.7)
            print(f'Gridman 🤖: {response}')
            if chat_over:
                break
            else:
                user_input = None