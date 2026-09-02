class ByteTokenizer:
    """
    字节 Tokenizer, 无需训练, 全世界通用
    """
    def __init__(self):
        # 基础特殊符号
        self.pad_token_id = 256 
        self.eos_token_id = 257   
        
        # 特殊符号
        self.system_token_id = 258
        self.user_token_id = 259
        self.assistant_token_id = 260

        # 预留部分
        self.vocab_size = 300

    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))

    def decode(self, ids: list[int]) -> str:
        clean_ids = [i for i in ids if 0 <= i < 256]
        return bytes(clean_ids).decode('utf-8', errors='replace')
    
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}