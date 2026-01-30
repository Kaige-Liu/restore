import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
snr = [0, 3, 6, 9, 12, 15, 18]
success_rate_normal = [0.98868687, 0.99626263, 0.99888889, 0.99974747, 1, 1, 1]
success_rate_attack1 = [0.99974747, 1, 1, 1, 1, 1, 1]
success_rate_attack2 = [0.98631313, 0.99833333, 0.99964646, 0.99994949, 1, 1, 1]

# 创建画布
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# 绘制折线图
line1, = ax.plot(snr, success_rate_normal, marker='o', color='black',
                linewidth=1.5, markersize=6, label='正向检测成功率')
line2, = ax.plot(snr, success_rate_attack1, marker='^', color='red',
                linewidth=1.5, markersize=6, label='第一类攻击检测成功率')
line3, = ax.plot(snr, success_rate_attack2, marker='s', color='blue',
                linewidth=1.5, markersize=6, label='第二类攻击检测成功率')

# 设置坐标轴范围
ax.set_xlim(min(snr)-1, max(snr)+1)
ax.set_ylim(0.98, 1.005)

# 设置坐标轴标签和标题
ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('检测成功率', fontsize=12)
ax.set_title('不同 SNR 下的检测成功率对比', fontsize=14, pad=10)

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 添加网格线
ax.grid(linestyle='--', color='lightgray', linewidth=0.8)

# 添加图例
legend = ax.legend(loc='lower right', fontsize=10, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')

# 确保中文显示正常
for text in legend.get_texts():
    text.set_fontname("SimHei")

# 设置y轴为百分比格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:.2%}"))

# 调整布局
plt.tight_layout()

# 保存图像
try:
    fig.savefig('detection_success_rate.png', dpi=300, bbox_inches='tight')
    print("图像已成功保存为 detection_success_rate.png")
except Exception as e:
    print(f"保存图像时出错: {e}")
    print("尝试使用替代方法保存...")
    try:
        canvas = fig.canvas
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.imsave('detection_success_rate_alt.png', image)
        print("图像已成功保存为 detection_success_rate_alt.png")
    except Exception as e2:
        print(f"替代方法也失败了: {e2}")
        print("请尝试在交互环境中显示图像或检查您的 matplotlib 配置")

# 显示图像
plt.show()