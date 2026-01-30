import json
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_and_save_loss_data(log_dir, scalar_names, output_file="loss_data_new.json"):
    """提取loss数据并保存为JSON文件"""
    # 提取数据
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    loss_data = {}
    for name in scalar_names:
        if name in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(name)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            loss_data[name] = {'steps': steps, 'values': values}
            print(f"成功提取 {name}：{len(steps)} 个数据点")
        else:
            print(f"警告：未找到标量 {name}")

    # 保存为JSON
    with open(output_file, 'w') as f:
        json.dump(loss_data, f, indent=2)

    print(f"数据已保存至 {output_file}")
    return loss_data


# 使用示例
log_dir = "C:\d\code\deepsc_mac_final\logs\deepsc_mac\\2025_12_10_01_17_06"  # 替换为实际日志目录
scalar_names = [
    'Loss_deepsc'
]

extract_and_save_loss_data(log_dir, scalar_names, "loss_data_new.json")