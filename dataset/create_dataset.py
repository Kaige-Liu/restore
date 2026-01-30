import os
import pickle
from PIL import Image

# 假设图片所在的文件夹路径
image_dir = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\dataset\MNIST_data\MNIST_32_train\\train"

# 获取文件夹中所有图片的文件名
image_files = os.listdir(image_dir)

# 创建一个空列表，用于存储图片数据
image_data = []
ct = 1
# 遍历图片文件，并将每张图片加载到内存中
for filename in image_files:
    # 构建图片文件的完整路径
    image_path = os.path.join(image_dir, filename)

    # 使用with语句打开图片文件，并在使用完毕后自动关闭
    with open(image_path, "rb") as f1:
        # 使用Pillow库加载图片
        image = Image.open(f1)

        image_data.append(image)
        print(ct)
        ct += 1


# 使用pickle将图片数据保存到文件中
output_file = "D:\deeplearning\deepsc_hiding_torch\deepsc_hiding_torch\data\image\hiding_data_dataset.pkl"
with open(output_file, "wb") as f:
    pickle.dump(image_data, f)

print("Image dataset saved to", output_file)
# 如果需要，可以在这里对图片进行一些预处理

# 将图片数据添加到列表中
