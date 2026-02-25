# import torch
# from PIL import Image
# from diffusers import StableDiffusionUpscalePipeline
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# pipe = StableDiffusionUpscalePipeline.from_pretrained(
#     "stabilityai/stable-diffusion-x4-upscaler",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# ).to(device)
#
# low_res = Image.open("sd_result.png").convert("RGB")
#
# prompt = "high quality photo, sharp details"
# image = pipe(prompt=prompt, image=low_res, num_inference_steps=30).images[0]
# image.save("upscaled.png")
# print("saved: upscaled.png")
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
from diffusers import UNet1DModel, DDIMScheduler
from tqdm import tqdm
import os
import datetime


# ==========================================
# 1. 归一化方案 A
# ==========================================
class FeatureScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, x):
        self.mean = x.mean()
        self.std = x.std()
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x_norm):
        return x_norm * self.std + self.mean


# ==========================================
# 2. 条件扩散模型类
# ==========================================
import torch.nn.functional as F


# 简单的残差块，用于处理 1D 特征
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class SemComDiffusion(nn.Module):
    def __init__(self, feat_dim=128, seq_len=31):
        super().__init__()
        self.feat_dim = feat_dim

        # 时间步嵌入：将标量 t 转为向量
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        # 这里的输入通道是 feat_dim * 2 (特征 + 条件)
        self.input_layer = nn.Conv1d(feat_dim * 2, 256, kernel_size=1)

        # 将时间步信息注入特征的层
        self.time_proj = nn.Linear(128, 256)

        # 核心卷积层
        self.net = nn.Sequential(
            ResBlock1D(256),
            ResBlock1D(256),
            ResBlock1D(256),
            nn.Conv1d(256, feat_dim, kernel_size=1)  # 输出维度恢复到 feat_dim
        )

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

    def forward(self, f_t, t, f_cond):
        """
        f_t: [bs, 128, 31]
        t: [bs] 或 int
        f_cond: [bs, 128, 31]
        """
        # 1. 拼接输入
        x = torch.cat([f_t, f_cond], dim=1)  # [bs, 256, 31]
        x = self.input_layer(x)

        # 2. 处理时间步
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x.device).float()
        else:
            t = t.float().view(-1, 1)

        t_emb = self.time_embed(t)  # [bs, 128]
        t_proj = self.time_proj(t_emb).unsqueeze(-1)  # [bs, 256, 1]

        # 3. 注入时间步信息并运行网络
        x = x + t_proj  # 广播相加
        return self.net(x)


# ==========================================
# 3. 主程序
# ==========================================
def run_experiment():
    # --- 初始化 TensorBoard ---
    writer = SummaryWriter(log_dir="./logs/34/" + "minitest")
    record_loss  =1000

    # --- 参数设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, seq_len, feat_dim = 64, 31, 128
    epochs = 1000
    lr = 1e-4

    # --- 模拟数据 (请替换为你的真实语义特征) ---
    f0_raw = torch.randn(2000, seq_len, feat_dim)
    f_cond_raw = f0_raw + 0.3 * torch.randn_like(f0_raw)  # 模拟受损特征

    # 划分训练集和验证集
    train_size = 1600
    scaler_f0 = FeatureScaler()
    scaler_cond = FeatureScaler()

    f0_norm = scaler_f0.fit_transform(f0_raw)
    f_cond_norm = scaler_cond.fit_transform(f_cond_raw)

    train_loader = DataLoader(TensorDataset(f0_norm[:train_size], f_cond_norm[:train_size]), batch_size=bs,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(f0_norm[train_size:], f_cond_norm[train_size:]), batch_size=bs, shuffle=False)

    # --- 模型初始化 ---
    model = SemComDiffusion(feat_dim=feat_dim, seq_len=seq_len).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- 训练循环 ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_f0, batch_cond in train_loader:
            batch_f0 = batch_f0.transpose(1, 2).to(device)
            batch_cond = batch_cond.transpose(1, 2).to(device)

            noise = torch.randn_like(batch_f0)
            timesteps = torch.randint(0, 1000, (batch_f0.shape[0],), device=device).long()
            noisy_f = model.scheduler.add_noise(batch_f0, noise, timesteps)

            noise_pred = model(noisy_f, timesteps, batch_cond)
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # --- 每 10 代进行一次验证集 Loss 计算 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_f0, v_cond in val_loader:
                v_f0, v_cond = v_f0.transpose(1, 2).to(device), v_cond.transpose(1, 2).to(device)
                v_noise = torch.randn_like(v_f0)
                v_timesteps = torch.randint(0, 1000, (v_f0.shape[0],), device=device).long()
                v_noisy_f = model.scheduler.add_noise(v_f0, v_noise, v_timesteps)
                v_pred = model(v_noisy_f, v_timesteps, v_cond)
                val_loss += criterion(v_pred, v_noise).item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- 保存模型 ---
        if avg_val_loss < record_loss:
            record_loss = avg_val_loss
            checkpoint = {
                "cdmodel": model.state_dict(),
            }
            torch.save(checkpoint, './checkpoints/34/' + now + '/checkpoint_{}'.format(epoch))
        writer.close()


if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    os.mkdir("./checkpoints/34/" + now)
    run_experiment()