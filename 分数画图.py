import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 新数据
snr = [0, 3, 6, 9, 12, 15, 18]
baseline = [0.59195, 0.842351, 0.88546, 0.89732, 0.89774, 0.897922, 0.89813]
proposed = [0.85297922, 0.93603073, 0.94502767, 0.94695773, 0.94761539, 0.9478321, 0.94803225]

# 创建画布
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# 绘制折线图
line1, = ax.plot(snr, baseline, marker='o', color='black',
                linewidth=1.5, markersize=6, label='Baseline')
line2, = ax.plot(snr, proposed, marker='^', color='red',
                linewidth=1.5, markersize=6, label='Proposed Scheme')

# 设置坐标轴范围
ax.set_xlim(-1, 20)  # 扩展x轴范围以包含20
ax.set_ylim(0.55, 1.0)  # 调整y轴范围以更好展示数据

# 设置坐标轴标签和标题
ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('BLEU SCORE', fontsize=12)  # 修改为BLEU Score
# ax.set_title('BLEU SCORE', fontsize=14, pad=10)

# 设置刻度字体大小和样式
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 设置x轴刻度为0, 5, 10, 15, 20
ax.set_xticks([0, 5, 10, 15, 20])

# 添加次要刻度（每个主刻度间有4个小刻度，间隔为1）
ax.xaxis.set_minor_locator(MultipleLocator(1))

# 设置y轴为三位小数格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:.3f}"))

# 添加网格线
ax.grid(linestyle='--', color='lightgray', linewidth=0.8)
ax.grid(which='minor', linestyle=':', color='lightgray', alpha=0.5)  # 添加次要网格线

# 添加图例
legend = ax.legend(loc='lower right', fontsize=10, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')

# 调整布局
plt.tight_layout()

# 保存图像
try:
    fig.savefig('comparison_bleu_score.png', dpi=300, bbox_inches='tight')
    print("Image saved successfully as comparison_bleu_score.png")
except Exception as e:
    print(f"Error saving image: {e}")
    print("Trying alternative method...")
    try:
        canvas = fig.canvas
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.imsave('comparison_bleu_score_alt.png', image)
        print("Image saved successfully as comparison_bleu_score_alt.png")
    except Exception as e2:
        print(f"Alternative method failed: {e2}")
        print("Please try displaying the image in an interactive environment")

# 显示图像
plt.show()