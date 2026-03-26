import torch
import torch.nn.functional as F

from grid_man import Gridman, Config
from ouro_dataloader import StreamLoader
from ouro_tools import save_checkpoint, load_checkpoint, print_model_parameters
    

if __name__ == "__main__":
    # 开启时只进行生成测试
    test_only= False
    config = Config()
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if test_only:
        grid_man = Gridman(config).to(device)
        load_checkpoint(grid_man, device=device)
        grid_man.eval()
        prompt_text = '太阳系是'
        
        prompt_ids = torch.tensor([config.tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
        
        with torch.amp.autocast('cuda', dtype=dtype):
            generated_ids = grid_man.generate(prompt_ids)
        
        generated_text = config.tokenizer.decode(generated_ids[0].tolist())
        print(f"Gridman 🤖: {generated_text}")
        print("-" * 65)

    else:
        # 标准预训练
        print("🚀 正在初始化预训练...")
        grid_man = Gridman(config).to(device)
        
        optimizer = torch.optim.AdamW(grid_man.parameters(), lr=3e-4)
        print_model_parameters(grid_man)
        
        dataloader = StreamLoader(config.patch_size, config.chunk_size, config.pretrain_train_file)
        tokenizer = config.tokenizer
        
        print("\n" + ">"*25 + " 开始极速流式预训练 " + "<"*25)

        loss_acc = torch.tensor(0.0, device=device)

        for step in range(200000): 
            grid_man.train()
            
            input_patches = dataloader.get_batch().to(device)
            
            # 构造 Input 和 Target
            inputs = input_patches[:, :-1]   
            targets = input_patches[:, 1:]  

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = grid_man(inputs)
                # 计算交叉熵损失
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    targets.reshape(-1)
                )
            
            loss_acc = loss_acc + loss

            if step % config.bptt_size == 0:
                loss_acc = loss_acc / config.bptt_size
                loss_acc.backward()
                torch.nn.utils.clip_grad_norm_(grid_man.parameters(), 1.0)
                optimizer.step()
                grid_man.core_ouro.mem_detach()
                print(f"\n📌 Step {step+1} | Total Loss = {loss_acc.item():.4f}")
                loss_acc = torch.tensor(0.0, device=device)

            if (step+1) % 5000 == 0:
                save_checkpoint(grid_man)
                
                grid_man.eval()
                prompt_text = "关于我是否有自我意识我的回答是"
            
                prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
                
                with torch.amp.autocast('cuda', dtype=dtype):
                    generated_ids = grid_man.generate(prompt_ids)
                
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                print(f"Gridman 🤖: {generated_text}")
                print("-" * 65)
