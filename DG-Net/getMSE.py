import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def rmetrics(a,b):
    
    # 防止图片大小不一样报错
    resized_image = cv2.resize(a, (b.shape[1], b.shape[0]))
    a = resized_image
    
    #pnsr
    psnr = compare_psnr(a,b)
    # mse = np.mean(np.square(a-b))
    # psnr = 10. * np.log10(np.square(255.) / mse)

    #ssim
    ssim = compare_ssim(a,b,win_size=3,channel_axis=True)
    # SSIM函数中的win_size参数设置为一个较小的奇数值，以便在计算结构相似性时覆盖图像的局部区域。
    # 一般建议选择一个比较小的值，以确保在图像的不同区域都能得到适当的比较，同时避免在计算时引入太多的噪声或失真。
    # 常见的选择包括3x3、5x5或7x7的窗口大小。
    # 可以根据你的图像大小和特性来进行调整和尝试，以找到最适合的窗口大小。
    # 在实际应用中，通常会根据图像的特点进行调整和优化。
    
    return psnr, ssim

def compute_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

# 文件夹路径
folder_U45 = './results_RCG/noise-300/noise-U45/res-U45-gaussian-5'
# folder_U45 = './results Weight2_3_2_0.2out_0.24Edge_0.1L_MSE+1.2Edge+5e-6VGG+12e-4FFL/noise-U45/res-U45-rayleigh-5'
folder_GT = './results_RCG/results_300/res-U45'
# folder_GT = './Test/Test-U45'

# 原退化图像和加噪版本
# gaussian-5    PSNR:  34.2834506161673   , SSIM:  0.9665431403822077   MSE: 24.2913
# gaussian-15   PSNR:  24.913415789709212 , SSIM:  0.830673985792582    MSE: 210.3564
# gaussian-30   PSNR:  19.06595290817774  , SSIM:  0.6656133145314005   MSE: 809.7100
# poisson       PSNR:  24.4906934744384   , SSIM:  0.8755168429843743   MSE: 424.6712
# rayleigh-5    PSNR:  31.719694500049677 , SSIM:  0.9811927169630034   MSE: 43.7707
# rayleigh-15   PSNR:  21.855199442700545 , SSIM:  0.8834521194023782   MSE: 424.4369
# rayleigh-30   PSNR:  15.873298072725493 , SSIM:  0.7291794738274852   MSE: 1684.3007  

# 匹配所有图像
img_paths = sorted(glob(os.path.join(folder_U45, '*')))
mse_list = []
psnr_list = []
ssim_list = []
sum = 0
print(folder_U45)
for path_U45 in img_paths:
    filename = os.path.basename(path_U45)
    path_GT = os.path.join(folder_GT, filename)
    sum += 1
    if not os.path.exists(path_GT):
        print(f"Warning: {filename} not found in {folder_GT}")
        continue
    print(sum, ": ", filename)

    # 读取图像（BGR模式）
    img_U45 = cv2.imread(path_U45)
    img_GT = cv2.imread(path_GT)

    # 检查尺寸是否一致
    if img_U45.shape != img_GT.shape:
        print(f"Warning: size mismatch for {filename}")
        continue

    mse = compute_mse(img_U45, img_GT)
    psnr, ssim = rmetrics(img_U45, img_GT)
    mse_list.append(mse)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

# 计算MSE均值
if mse_list:
    mean_mse = np.mean(mse_list)
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    print(f"Average MSE over {len(mse_list)} image pairs: {mean_mse:.4f}")
    print("PSNR: ", mean_psnr, ", SSIM: ", mean_ssim)
else:
    print("No valid image pairs found.")
