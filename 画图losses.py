import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


# -------------------------- 1. 读取 JSON 数据 --------------------------
def load_loss_data(json_path):
    """读取 loss_data.json，返回字典：{曲线名称: (epochs, losses)}"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for curve_name, curve_data in data.items():
        # 假设 JSON 结构：{"曲线1": {"steps": [1,2,...], "values": [0.1, 0.2,...]}}
        epochs = np.array(curve_data["steps"])
        losses = np.array(curve_data["values"])
        result[curve_name] = (epochs, losses)
    return result


# -------------------------- 2. 平滑处理函数 --------------------------
def smooth_loss(loss_values, window_size=51, poly_order=3):
    """
    使用 Savitzky-Golay 滤波器平滑曲线
    :param loss_values: 原始 Loss 数组
    :param window_size: 平滑窗口大小（需为奇数，建议 31~101）
    :param poly_order: 多项式拟合阶数（建议 2~4）
    :return: 平滑后的 Loss 数组
    """
    if len(loss_values) < window_size:
        return loss_values  # 数据量太少时不处理
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    return savgol_filter(loss_values, window_size, poly_order)


# -------------------------- 3. 论文风格绘图 --------------------------
def plot_paper_style_loss(loss_data, save_path="paper_loss_curve_new.png", window_size=51):
    """
    绘制论文级平滑 Loss 曲线
    :param loss_data: 字典，{曲线名称: (epochs, losses)}
    :param save_path: 图片保存路径
    :param window_size: 平滑窗口大小
    """
    plt.figure(figsize=(6, 4.5), dpi=300)  # 论文常用尺寸 + 高分辨率
    plt.style.use('default')  # 重置风格，避免默认样式干扰

    # 论文常用配色（可根据需求调整）
    color_map = {
        "Loss_deepsc": "black",
        "Loss_normal": "red",
        "Loss_bleu": "blue",
        # 可扩展更多曲线...
    }

    # 遍历每条曲线绘制
    for curve_name, (epochs, losses) in loss_data.items():
        # 平滑处理
        smoothed_loss = smooth_loss(losses, window_size=window_size)

        # 绘制曲线（论文风格：实线 + 清晰标记）
        plt.plot(
            epochs,
            smoothed_loss,
            label=curve_name,
            color=color_map.get(curve_name, "gray"),
            linewidth=1.5,
            linestyle="-"
        )

    # -------------------------- 坐标轴与标注 --------------------------
    plt.xlabel("Epoch", fontsize=12, fontfamily="Times New Roman")
    plt.ylabel("Loss", fontsize=12, fontfamily="Times New Roman")
    plt.xticks(fontsize=10, fontfamily="Times New Roman")
    plt.yticks(fontsize=10, fontfamily="Times New Roman")

    # 可根据数据范围调整坐标轴
    # plt.xlim(0, max(epochs))
    # plt.ylim(0, max(losses)*1.1)

    # -------------------------- 图例设置（论文风格） --------------------------
    legend = plt.legend(
        loc="upper right",  # 可根据曲线位置调整
        fontsize=9,
        frameon=True,  # 显示图例边框
        facecolor="white",  # 图例背景白色
        edgecolor="black"  # 图例边框黑色
    )
    # 强制图例文字用 Times New Roman
    for text in legend.get_texts():
        text.set_fontfamily("Times New Roman")

    # -------------------------- 网格线 --------------------------
    plt.grid(linestyle="--", color="lightgray", linewidth=0.8)

    # -------------------------- 保存 & 显示 --------------------------
    plt.tight_layout()  # 自动优化布局
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"论文风格曲线已保存至：{save_path}")



# -------------------------- 4. 主流程调用 --------------------------
if __name__ == "__main__":
    # 替换为你的 JSON 文件路径
    json_path = "loss_data_new.json"
    # 读取数据
    loss_data = load_loss_data(json_path)

    # 绘制曲线（可调整 window_size 控制平滑程度）
    plot_paper_style_loss(
        loss_data,
        save_path="paper_style_loss_new.png",
        window_size=51  # 窗口越大越平滑，建议 31~101
    )