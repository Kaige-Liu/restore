import tensorflow as tf
import matplotlib.pyplot as plt

# 读取events文件
event_file = "C:\d\code\deepsc_mac_final\logs\deepsc_mac\\2025_12_10_01_17_06\events.out.tfevents.1765300626.autodl-container-a03b4fb96c-2f263b56.5257.0"  # 比如 "./events.out.tfevents.1733800000.autodl-container"
loss_list = []
step_list = []

for event in tf.compat.v1.train.summary_iterator(event_file):
    for value in event.summary.value:
        # 替换成你实际的loss标签名（比如"loss"、"train_loss"）
        if value.tag == "loss":
            loss_list.append(value.simple_value)
            step_list.append(event.step)

# 画loss曲线
plt.plot(step_list, loss_list, label="Training Loss", color="red")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()