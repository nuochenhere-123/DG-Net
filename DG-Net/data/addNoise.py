import os
import cv2
import numpy as np
from glob import glob
import random

# 添加高斯噪声
def add_gaussian_noise(image, std=25):
    """
    参数说明：
        mean：高斯分布的均值，通常设为0
        std：标准差，值越大噪声越强，图像越模糊/颗粒感越重
    """
    mean = 0
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_img = np.clip(image + noise, 0, 255)  # 加噪并限制像素范围
    return noisy_img.astype(np.uint8)

# 添加泊松噪声（基于图像强度的自然噪声）
def add_poisson_noise(image, mark = 0):
    """
    泊松噪声本质上依赖于图像的像素值本身，强度较高区域会有更大的波动。
    通常不需要额外参数，模拟感光元件的随机性。
    """
    noisy_img = np.random.poisson(image.astype(np.float32)).astype(np.uint8)
    return np.clip(noisy_img, 0, 255)

# 添加瑞利噪声  输入[0, 255]，类型是 uint8；输出[0, 255]，类型是 uint8
def add_rayleigh_noise(image, scale=25):
    """
    参数说明：
        scale：瑞利分布的尺度参数，值越大，添加的噪声强度越大
        通常模拟水下、雷达成像等场景中的乘性噪声。
    """
    noise = np.random.rayleigh(scale, image.shape).astype(np.float32)
    noisy_img = np.clip(image + noise, 0, 255)
    return noisy_img.astype(np.uint8)


# 随机选择噪声类型并应用
def getRandomNoise(img_np):
    """
    随机选择一种噪声并返回加噪后的图像
    参数：
        img: 输入图像，PIL 图像格式
    输出：
        加噪后的图像，PIL 图像格式
    """
    # 输入为 numpy数组 0-255 uint8 或 float32
    
    # 随机选择噪声类型
    noise_type = random.choice(['gaussian', 'poisson', 'rayleigh'])
    
    if noise_type == 'gaussian':
        std = random.randint(15, 35)  # 随机标准差
        noisy_img = add_gaussian_noise(img_np, std = std)
    
    elif noise_type == 'poisson':
        noisy_img = add_poisson_noise(img_np, mark = 0)
    
    elif noise_type == 'rayleigh':
        scale = random.randint(15, 35)  # 随机尺度参数
        noisy_img = add_rayleigh_noise(img_np, scale = scale)

    # 输出[0, 255]，类型是 uint8
    return noisy_img


# # 主处理函数：批量读取图像、加噪并保存
# def process_images(input_dir='./TestingSet/Test-OceanDark'):
#     dataset = 'OceanDark'
#     scale_sum = [5, 15, 30]
#     # 查找所有png/jpg/jPEG格式的图像
#     image_paths = glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, f'*.jpg')) + glob(os.path.join(input_dir, f'*.jpeg'))
    
#     # 定义可用的噪声类型及其对应的处理函数
#     noise_types = {
#         'gaussian': add_gaussian_noise,
#         'poisson': add_poisson_noise,
#         'rayleigh': add_rayleigh_noise,
#     }

#     # 不同程度噪声
#     now = 0
#     for scale in scale_sum:
#         now += 1
#         # 逐类处理噪声
#         for noise_name, noise_func in noise_types.items():
#             if now != 1 and noise_name == 'poisson':
#                 continue
#             elif noise_name == 'poisson':
#                 output_dir = f'./noise-{dataset}/{dataset}-{noise_name}'
#             else:
#                 output_dir = f'./noise-{dataset}/{dataset}-{noise_name}-{scale}'
#             os.makedirs(output_dir, exist_ok=True)
#             print(f'Processing {noise_name} noise...')

#             for img_path in image_paths:
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     continue
                
#                 # 调用对应的噪声函数
#                 noisy_img = noise_func(img, scale)
                
#                 # 保存加噪图像
#                 filename = os.path.basename(img_path)
#                 save_path = os.path.join(output_dir, filename)
#                 cv2.imwrite(save_path, noisy_img)
            
#             print(f'Saved to {output_dir}')

# if __name__ == "__main__":
#     process_images()
