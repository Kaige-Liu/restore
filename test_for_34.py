import os

# 1. 镜像站，确保在服务器上能下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import StableDiffusionPipeline

# 2. 使用最标准的 SD v1.5 模型
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"

print(f"🚀 正在从镜像站加载 Stable Diffusion 模型...")

try:
    # 3. 加载全家桶 (Pipeline 会自动处理所有配置文件)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16  # 使用半精度，速度翻倍，显存减半
    ).to(device)

    # 4. 生成一张图
    # prompt 是你想要画的内容
    # num_inference_steps=30 就是在使用类似 DDIM 的高效采样
    prompt = "a beautiful landscape painting, highly detailed, oil on canvas"

    print("🎨 正在生成图像，请稍候...")
    image = pipe(prompt, num_inference_steps=30).images[0]

    # 5. 保存结果
    image.save("sd_result.png")
    print("\n✨ 成功！图片已保存为 sd_result.png")

except Exception as e:
    print(f"\n❌ 出错了: {e}")