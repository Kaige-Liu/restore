import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams['figure.dpi'] = 300

# 数据
snr = [0, 3, 6, 9, 12, 15, 18]
success_rate_normal = [0.98868687, 0.99626263, 0.99888889, 0.99974747, 1, 1, 1]
success_rate_attack1 = [0.99974747, 1, 1, 1, 1, 1, 1]
success_rate_attack2 = [0.98631313, 0.99833333, 0.99964646, 0.99994949, 1, 1, 1]

# 创建画布
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# 绘制折线图
line1, = ax.plot(snr, success_rate_normal, marker='o', color='black',
                linewidth=1.5, markersize=6, label='Normal Detection Success Rate')
line2, = ax.plot(snr, success_rate_attack1, marker='^', color='red',
                linewidth=1.5, markersize=6, label='Type I Attack Detection Success Rate')
line3, = ax.plot(snr, success_rate_attack2, marker='s', color='blue',
                linewidth=1.5, markersize=6, label='Type II Attack Detection Success Rate')

# 设置坐标轴范围
ax.set_xlim(min(snr)-1, max(snr)+1)
ax.set_ylim(0.98, 1.005)
ax.set_xticks([0, 5, 10, 15, 20])
# 设置坐标轴标签和标题
ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Detection Success Rate', fontsize=12)
# ax.set_title('Detection Success Rate vs. SNR', fontsize=14, pad=10, fontname='Times New Roman')

# 设置刻度字体大小和样式
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 添加网格线
ax.grid(linestyle='--', color='lightgray', linewidth=0.8)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.grid(which='minor', linestyle=':', color='lightgray', alpha=0.5)  # 添加次要网格线

# 添加图例
legend = ax.legend(loc='lower right', fontsize=10, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')

# 设置y轴为百分比格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:.2%}"))

# 调整布局
plt.tight_layout()

# 保存图像
try:
    fig.savefig('detection_success_rate.png', dpi=300, bbox_inches='tight')
    print("Image saved successfully as detection_success_rate.png")
except Exception as e:
    print(f"Error saving image: {e}")
    print("Trying alternative method...")
    try:
        canvas = fig.canvas
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.imsave('detection_success_rate_alt.png', image)
        print("Image saved successfully as detection_success_rate_alt.png")
    except Exception as e2:
        print(f"Alternative method failed: {e2}")
        print("Please try displaying the image in an interactive environment")

# 显示图像
plt.show()