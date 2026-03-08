import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_compare(clean, noisy, denoised, idx=0, save_path=None):
    a = clean[idx].detach().cpu().numpy()
    b = noisy[idx].detach().cpu().numpy()
    c = denoised[idx].detach().cpu().numpy()

    vmin = min(a.min(), b.min(), c.min())
    vmax = max(a.max(), b.max(), c.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # 先给右边留出 colorbar 的位置
    fig.subplots_adjust(right=0.88, wspace=0.30)

    titles = ['Original Tensor', 'Noisy Tensor', 'Denoised Tensor']
    data_list = [a, b, c]

    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            cmap='magma',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Position')
        ax.set_ylabel('Channel')

    # 单独给 colorbar 开一个轴: [left, bottom, width, height]
    cax = fig.add_axes([0.90, 0.18, 0.015, 0.68])
    fig.colorbar(im, cax=cax)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)   # 这里先别用 bbox_inches='tight'
    plt.close(fig)




import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

def plot_diffusion_paper_style(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    idx: int = 0,
    save_path: str = 'moren.png'
):
    """
    更适合论文排版的版本：
    目标语义 / 受损语义 / 恢复语义 / |受损语义-目标语义| / |恢复语义-目标语义|
    """

    font_path = "/root/autodl-tmp/restore/fonts/STSONG.TTF"
    font_prop = font_manager.FontProperties(fname=font_path)

    # 不要再设置 font.family = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False

    x_clean = clean[idx].detach().cpu().numpy()
    x_noisy = noisy[idx].detach().cpu().numpy()
    x_denoised = denoised[idx].detach().cpu().numpy()

    err_noisy = np.abs(x_noisy - x_clean)
    err_denoised = np.abs(x_denoised - x_clean)

    vmin_main = min(x_clean.min(), x_noisy.min(), x_denoised.min())
    vmax_main = max(x_clean.max(), x_noisy.max(), x_denoised.max())
    vmax_err = max(err_noisy.max(), err_denoised.max())

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.subplots_adjust(right=0.92, wspace=0.22)

    # 前三张主图
    titles_main = ['目标语义', '受损语义', '恢复语义']
    data_main = [x_clean, x_noisy, x_denoised]

    for i in range(3):
        im_main = axes[i].imshow(
            data_main[i],
            aspect='auto',
            origin='lower',
            cmap='magma',
            vmin=vmin_main,
            vmax=vmax_main
        )
        axes[i].set_title(
            titles_main[i],
            fontsize=12,
            fontproperties=font_prop
        )
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # 第4张：受损误差
    im_err1 = axes[3].imshow(
        err_noisy,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=vmax_err
    )
    axes[3].set_title(
        '|受损语义-目标语义|',
        fontsize=12,
        fontproperties=font_prop
    )
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    # 第5张：恢复误差
    im_err2 = axes[4].imshow(
        err_denoised,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=vmax_err
    )
    axes[4].set_title(
        '|恢复语义-目标语义|',
        fontsize=12,
        fontproperties=font_prop
    )
    axes[4].set_xticks([])
    axes[4].set_yticks([])

    # 两个 colorbar
    cax1 = fig.add_axes([0.935, 0.55, 0.012, 0.28])
    fig.colorbar(im_main, cax=cax1)

    cax2 = fig.add_axes([0.935, 0.15, 0.012, 0.28])
    fig.colorbar(im_err2, cax=cax2)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)