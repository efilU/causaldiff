import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

# ---------------------------------------------------
# 1. 从您项目根目录的 choices.py 文件导入所有定义
#    (不再在此文件中重复定义)
# ---------------------------------------------------
from choices import *

# ---------------------------------------------------
# 2. 从您的文件夹结构中导入模型和流程模块
# ---------------------------------------------------
from model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from diffusion.diffusion import SpacedDiffusionBeatGans, SpacedDiffusionBeatGansConfig, space_timesteps
from diffusion.base import get_named_beta_schedule

# ---------------------------------------------------
# 3. 训练配置和超参数
# ---------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")
DATA_DIR = "data/"
BATCH_SIZE = 4  # 为GTX 1650设置的小批量
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_PRETRAIN_EPOCHS = 1 # 快速检验轮次
NUM_TRAIN_EPOCHS = 1    # 快速检验轮次
PRETRAIN_KLD_WEIGHT = 1e-5
TRAIN_RECON_WEIGHT = 1.0
TRAIN_CLS_WEIGHT = 1e-2
TRAIN_CLUB_WEIGHT = 1e-5
TRAIN_KLD_WEIGHT = 1e-5

# ---------------------------------------------------
# 4. 数据加载函数
# ---------------------------------------------------
def get_cifar10_dataloader(batch_size, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    # 在Windows上，num_workers通常建议设为0以避免多进程问题，您可以尝试设为4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return dataloader

# ---------------------------------------------------
# 5. 预训练和联合训练函数
# ---------------------------------------------------
def pretrain_causal_diff(model, diffusion_process, dataloader, epochs, device):
    print("\n--- [阶段1] 开始预训练 (快速检验模式) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        for step, (x_start, _) in enumerate(dataloader):
            if step >= 20:
                print("--- 预训练快速检验完成20步，退出循环 ---")
                break
            optimizer.zero_grad()
            x_start = x_start.to(device)
            t = torch.randint(0, diffusion_process.num_timesteps, (x_start.shape[0],), device=device).long()
            losses = diffusion_process.training_losses(model, x_start, t)
            pretrain_loss = (losses["mse"].mean() + PRETRAIN_KLD_WEIGHT * losses["kld_loss"].mean())
            pretrain_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f"[预训练检验 Step {step+1}/20] Loss: {pretrain_loss.item():.4f}")
    print("--- [阶段1] 预训练检验完成 ---")

def train_causal_diff(model, diffusion_process, dataloader, epochs, device):
    print("\n--- [阶段2] 开始联合训练 (快速检验模式) ---")
    main_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if model.conf.use_club:
        club_optimizer = optim.Adam(model.club.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        for step, (x_start, target_y) in enumerate(dataloader):
            if step >= 20:
                print("--- 联合训练快速检验完成20步，退出循环 ---")
                break
            main_optimizer.zero_grad()
            x_start = x_start.to(device)
            target_y = target_y.to(device)
            t = torch.randint(0, diffusion_process.num_timesteps, (x_start.shape[0],), device=device).long()
            
            losses = diffusion_process.training_losses(model, x_start, t, target_y=target_y)
            
            # !!! 核心修正部分: 移除了对已经是标量的损失项的 .mean() 调用 !!!
            total_loss = (
                TRAIN_RECON_WEIGHT * losses["mse"].mean() +       # mse是按样本的，需要.mean()
                TRAIN_KLD_WEIGHT * losses["kld_loss"].mean() +   # kld_loss是按样本的，需要.mean()
                TRAIN_CLS_WEIGHT * losses["cls_loss"]            # cls_loss已经是平均值，无需.mean()
            )
            if model.conf.use_club:
                # club_loss也已经是平均值，无需.mean()
                total_loss += TRAIN_CLUB_WEIGHT * losses["club_loss"]

            total_loss.backward(retain_graph=model.conf.use_club)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            main_optimizer.step()
            
            if model.conf.use_club:
                club_optimizer.zero_grad()
                with torch.no_grad():
                    cond = model.encode(x_start)['cond']
                    z = cond[:, :model.lacim.z_dim]; s = cond[:, model.lacim.s_dim:]
                club_learning_loss = model.club.learning_loss(z, s)
                club_learning_loss.backward(); club_optimizer.step()
            
            # !!! 核心修正部分: 对已经是标量的损失，不再调用 .item() !!!
            club_val = losses["club_loss"] if model.conf.use_club else 0
            # 如果 club_val 是 tensor, 才调用 .item()
            if isinstance(club_val, torch.Tensor):
                club_val = club_val.item()
                
            cls_loss_val = losses['cls_loss']
            if isinstance(cls_loss_val, torch.Tensor):
                cls_loss_val = cls_loss_val.item()
                
            print(f"[联合训练检验 Step {step+1}/20] Total Loss: {total_loss.item():.4f} | "
                  f"MSE: {losses['mse'].mean().item():.4f} | "
                  f"Cls: {cls_loss_val:.4f} | "
                  f"KLD: {losses['kld_loss'].mean().item():.4f} | "
                  f"CLUB: {club_val:.4f}")
                  
    print("--- [阶段2] 联合训练检验完成 ---")

# ---------------------------------------------------
# 6. 主程序入口
# ---------------------------------------------------
if __name__ == '__main__':
    # ---- 1. 初始化模型架构配置 ----
    autoenc_config = BeatGANsAutoencConfig(
        # --- BeatGANsUNetConfig 基础参数 ---
        image_size=IMAGE_SIZE,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16, 8), # 为32x32图像调整
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        dims=2,
        num_classes=None, # 保持为None以避免之前的NotImplementedError
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=-1,
        resblock_updown=True,
        use_new_attention_order=False,
        resnet_two_cond=True, # 确保使用双条件分支

        # --- BeatGANsAutoencConfig 扩展参数 ---
        enc_out_channels=512,
        use_club=True,
        club_hidden_dim=256
    )
    
    # ---- 2. 初始化扩散流程配置 ----
    betas = get_named_beta_schedule("linear", 1000)
    use_timesteps = space_timesteps(1000, "1000")
    
    diffusion_config = SpacedDiffusionBeatGansConfig(
        # ---- 使用从 choices.py 导入的正确枚举类型 ----
        gen_type=GenerativeType.ddpm,
        # !!! 关键修正: 根据您提供的 choices.py，正确的模型类型是 autoencoder !!!
        model_type=ModelType.autoencoder, 
        loss_type=LossType.mse,
        model_mean_type=ModelMeanType.eps,
        model_var_type=ModelVarType.fixed_small,
        # ---- 其他参数 ----
        fp16=False,
        betas=betas,
        rescale_timesteps=True,
        use_timesteps=use_timesteps,
        # !!! 关键修正: 为 mask_threshold 提供一个具体的浮点数值 !!!
        mask_threshold=0.1,
        # !!! 关键修正: 同样需要在此处启用use_club，以通知训练流程计算club_loss !!!
        use_club=True
    )

    # ---- 3. 创建模型和流程管理器实例 ----
    causal_diff_model = BeatGANsAutoencModel(autoenc_config).to(DEVICE)
    diffusion_process_manager = SpacedDiffusionBeatGans(diffusion_config)

    # ---- 4. 加载数据集 ----
    cifar10_loader = get_cifar10_dataloader(BATCH_SIZE, DATA_DIR)
    
    # ---- 5. 按顺序执行训练流程 ----
    pretrain_causal_diff(
        model=causal_diff_model,
        diffusion_process=diffusion_process_manager,
        dataloader=cifar10_loader,
        epochs=NUM_PRETRAIN_EPOCHS,
        device=DEVICE
    )
    train_causal_diff(
        model=causal_diff_model,
        diffusion_process=diffusion_process_manager,
        dataloader=cifar10_loader,
        epochs=NUM_TRAIN_EPOCHS,
        device=DEVICE
    )
    
    # ---- 6. 保存最终模型 ----
    print("\nCausalDiff 完整训练流程完成！正在保存最终模型...")
    torch.save(causal_diff_model.state_dict(), "causal_diff_cifar10_final.pt")
    print("模型已保存至 causal_diff_cifar10_final.pt")