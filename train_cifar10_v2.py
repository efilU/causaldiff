import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import torch.nn.functional as F

from choices import *
from model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from diffusion.diffusion import SpacedDiffusionBeatGans, SpacedDiffusionBeatGansConfig, space_timesteps
from diffusion.base_v2 import get_named_beta_schedule, mean_flat # 导入mean_flat

# --- 训练配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用的设备: {DEVICE}")
DATA_DIR = "data/"
BATCH_SIZE = 4
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_PRETRAIN_EPOCHS = 1
NUM_TRAIN_EPOCHS = 1
PRETRAIN_KLD_WEIGHT = 1e-5
TRAIN_RECON_WEIGHT = 1.0
TRAIN_CLS_WEIGHT = 1e-2
TRAIN_CLUB_WEIGHT = 1e-5
TRAIN_KLD_WEIGHT = 1e-5

# --- 数据加载 ---
def get_cifar10_dataloader(batch_size, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return dataloader

# --- 训练函数 (重构后) ---
def pretrain_causal_diff(model, diffusion_process, dataloader, epochs, device):
    print("\n--- [阶段1] 开始预训练 (重构版) ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        for step, (x_start, _) in enumerate(dataloader):
            if step >= 20: break
            optimizer.zero_grad()
            x_start = x_start.to(device)
            t = torch.randint(0, diffusion_process.num_timesteps, (x_start.shape[0],), device=device).long()
            noise = torch.randn_like(x_start)
            x_t = diffusion_process.q_sample(x_start, t, noise=noise)

            # 1. 编码
            tmp = model.encode(x_start)
            cond, mu, logvar = tmp['cond'], tmp['mu'], tmp['logvar']
            
            # 2. 计算KLD损失
            kld_loss = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=1).mean()

            # 3. 条件丢弃
            mask = torch.rand(x_start.size(0), device=device) < diffusion_process.conf.mask_threshold
            cond_gen = cond.clone()
            cond_gen[mask] = 0.

            # 4. U-Net预测
            predicted_noise = model(x_t, t, x_start=x_start, cond=cond_gen).pred

            # 5. 计算重建损失
            recon_loss = mean_flat((noise - predicted_noise)**2).mean()

            # 6. 组合预训练损失并更新
            pretrain_loss = recon_loss + PRETRAIN_KLD_WEIGHT * kld_loss
            pretrain_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            print(f"[预训练检验 Step {step+1}/20] Loss: {pretrain_loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | KLD: {kld_loss.item():.4f}")
    print("--- [阶段1] 预训练检验完成 ---")


def train_causal_diff(model, diffusion_process, dataloader, epochs, device):
    print("\n--- [阶段2] 开始联合训练 (重构版) ---")
    main_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if model.conf.use_club:
        club_optimizer = optim.Adam(model.club.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        for step, (x_start, target_y) in enumerate(dataloader):
            if step >= 20: break
            main_optimizer.zero_grad()
            x_start, target_y = x_start.to(device), target_y.to(device)
            t = torch.randint(0, diffusion_process.num_timesteps, (x_start.shape[0],), device=device).long()
            noise = torch.randn_like(x_start)
            x_t = diffusion_process.q_sample(x_start, t, noise=noise)

            # --- CIB损失计算 ---
            # 1. 编码 (获取s, z, mu, logvar)
            tmp = model.encode(x_start)
            cond, mu, logvar = tmp['cond'], tmp['mu'], tmp['logvar']
            s = cond[:, model.lacim.s_dim:]
            z = cond[:, :model.lacim.z_dim]

            # 2. 计算分类损失
            pred_y_log_softmax = model.dec_y(s)
            cls_loss = F.nll_loss(pred_y_log_softmax, target_y)

            # 3. 计算KLD损失
            kld_loss = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=1).mean()

            # 4. 计算CLUB损失
            club_loss = torch.tensor(0.0, device=device)
            if model.conf.use_club:
                club_loss = model.club(z, s)

            # 5. 条件丢弃和U-Net预测
            mask = torch.rand(x_start.size(0), device=device) < diffusion_process.conf.mask_threshold
            cond_gen = cond.clone()
            cond_gen[mask] = 0.
            predicted_noise = model(x_t, t, x_start=x_start, cond=cond_gen).pred

            # 6. 计算重建损失
            recon_loss = mean_flat((noise - predicted_noise)**2).mean()

            # 7. 组合最终CIB损失
            total_loss = (TRAIN_RECON_WEIGHT * recon_loss +
                          TRAIN_CLS_WEIGHT * cls_loss +
                          TRAIN_KLD_WEIGHT * kld_loss +
                          TRAIN_CLUB_WEIGHT * club_loss)

            total_loss.backward(retain_graph=model.conf.use_club)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            main_optimizer.step()

            # --- CLUB独立更新 ---
            if model.conf.use_club:
                club_optimizer.zero_grad()
                # 使用无梯度的s,z来更新CLUB估算器
                club_learning_loss = model.club.learning_loss(z.detach(), s.detach())
                club_learning_loss.backward()
                club_optimizer.step()

            print(f"[联合训练检验 Step {step+1}/20] Total: {total_loss.item():.4f} | "
                  f"MSE: {recon_loss.item():.4f} | Cls: {cls_loss.item():.4f} | "
                  f"KLD: {kld_loss.item():.4f} | CLUB: {club_loss.item():.4f}")
    print("--- [阶段2] 联合训练检验完成 ---")

# --- 主程序入口 ---
if __name__ == '__main__':
    autoenc_config = BeatGANsAutoencConfig(
        image_size=IMAGE_SIZE, in_channels=3, model_channels=128, out_channels=3,
        num_res_blocks=2, attention_resolutions=(16, 8), dropout=0.1,
        channel_mult=(1, 2, 2, 2), conv_resample=True, dims=2, num_classes=None,
        use_checkpoint=False, num_heads=4, num_head_channels=-1, resblock_updown=True,
        use_new_attention_order=False, resnet_two_cond=True, enc_out_channels=512,
        use_club=True, club_hidden_dim=256
    )
    betas = get_named_beta_schedule("linear", 1000)
    use_timesteps = space_timesteps(1000, "1000")
    diffusion_config = SpacedDiffusionBeatGansConfig(
        gen_type=GenerativeType.ddpm, model_type=ModelType.autoencoder,
        loss_type=LossType.mse, model_mean_type=ModelMeanType.eps,
        model_var_type=ModelVarType.fixed_small, fp16=False, betas=betas,
        rescale_timesteps=True, use_timesteps=use_timesteps,
        mask_threshold=0.1, use_club=True
    )
    causal_diff_model = BeatGANsAutoencModel(autoenc_config).to(DEVICE)
    diffusion_process_manager = SpacedDiffusionBeatGans(diffusion_config)
    cifar10_loader = get_cifar10_dataloader(BATCH_SIZE, DATA_DIR)
    
    pretrain_causal_diff(model=causal_diff_model, diffusion_process=diffusion_process_manager,
                         dataloader=cifar10_loader, epochs=NUM_PRETRAIN_EPOCHS, device=DEVICE)
    train_causal_diff(model=causal_diff_model, diffusion_process=diffusion_process_manager,
                        dataloader=cifar10_loader, epochs=NUM_TRAIN_EPOCHS, device=DEVICE)
    
    print("\nCausalDiff 完整训练流程完成！正在保存最终模型...")
    torch.save(causal_diff_model.state_dict(), "causal_diff_cifar10_final.pt")
    print("模型已保存至 causal_diff_cifar10_final.pt")