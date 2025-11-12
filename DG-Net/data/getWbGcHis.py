import os
import glob
import matplotlib.pyplot as plt
import shutil
from PIL import Image  # for loading images as YCbCr format
import numpy as np
import os
import skimage.io as io
import cv2

# 读取单张图片
def get_image_original(image_path,is_grayscale=False):
  image = io.imread(image_path, is_grayscale)
  image = image.astype(np.float32)
  if image.shape[-1] == 4:
    # 如果图像有四个通道，假设第四个通道为 alpha 通道，只保留前三个通道（RGB）
    image = image[:, :, :3]
  return image/255.0


def apply_white_balance(img):
    img = np.clip((img * 255.0).astype(np.uint8), 0, 255)
    # 简单灰度世界算法（Gray World Assumption）
    result = cv2.xphoto.createSimpleWB()
    return  np.clip(result.balanceWhite(img)/255.0, 0, 1)


# 获取白平衡结果
def get_images_wb(img):
    im_rgb = (img*255).astype(np.uint8)
    # if RGB
    R = np.sum(im_rgb[:, :, 0], axis=None)
    G = np.sum(im_rgb[:, :, 1], axis=None)
    B = np.sum(im_rgb[:, :, 2], axis=None)

    maxpix = max(R, G, B)
    if maxpix < 1e-6:
        maxpix = 1e-6  # 避免除以0
    ratio = np.array([R / maxpix, G / maxpix, B / maxpix])

    satLevel1 = 0.005 * ratio
    satLevel2 = 0.005 * ratio

    m, n, p = im_rgb.shape
    im_rgb_flat = np.zeros(shape=(p, m * n))
    for i in range(0, p):
        im_rgb_flat[i, :] = np.reshape(im_rgb[:, :, i], (1, m * n))

    wb = np.zeros(shape=im_rgb_flat.shape)
    for ch in range(p):
        q = [np.clip(satLevel1[ch], 0, 1), np.clip(1 - satLevel2[ch], 0, 1)]
        tiles = np.quantile(im_rgb_flat[ch, :], q)
        temp = im_rgb_flat[ch, :]
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        wb[ch, :] = temp
        bottom = min(wb[ch, :])
        top = max(wb[ch, :])
        # print(f"Channel {ch}: top={top}, bottom={bottom}")
        wb[ch, :] = (wb[ch, :] - bottom) * 255 / (max(top - bottom, 1e-6))

    outval = np.zeros(shape=im_rgb.shape)
    for i in range(p):
        outval[:, :, i] = np.reshape(wb[i, :], (m, n))
    
    outval = np.clip(outval/255.0, 0, 1)
    return outval


# 获取伽马校正结果
def get_images_gc(img, gamma=0.7):
  img_gc = np.power(img, gamma)
  img_gc = np.clip(img_gc, 0, 1)
  return img_gc


# 获取分块 (8,8) 直方图均衡化结果
def get_images_histeq(img):
  img = (img*255).astype(np.uint8)
  im_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
  clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
  el = clahe.apply(im_lab[:, :, 0])
  im_he = im_lab.copy()
  im_he[:, :, 0] = el
  img_histeq = cv2.cvtColor(im_he, cv2.COLOR_LAB2RGB)
  img_histeq = np.clip(img_histeq/255.0, 0, 1)
  return img_histeq


def get_his_after_cp(image):
    """
    传入归一化到[0,1]范围的图像numpy数组 (H, W, 3)，执行弱通道补偿 + 直方图拉伸，
    返回处理后的图像 (H, W, 3)，范围仍是[0,1]。
    """

    # 1. 弱通道补偿
    mean_r, mean_g, mean_b = np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])
    means = np.array([mean_r, mean_g, mean_b])  # RGB顺序
    max_idx = np.argmax(means)
    sorted_indices = np.argsort(means)
    min_idx, middle_idx = sorted_indices[0], sorted_indices[1]

    new_image = image.copy()
    for idx in [min_idx, middle_idx]:
        a = image[:, :, idx]
        b = image[:, :, max_idx]
        new_image[:, :, idx] = a + (means[max_idx] - means[idx]) * (1 - a) * b

    # 保证数值在[0,1]
    new_image = np.clip(new_image, 0, 1)


    # 2. 直方图拉伸（这里用0.8%和99.2%分位数）
    T1 = np.percentile(new_image, 0.8, axis=(0, 1))
    T2 = np.percentile(new_image, 99.2, axis=(0, 1))

    stretched_image = np.zeros_like(new_image)
    for c in range(3):
        channel = new_image[:, :, c]
        mask_low = channel <= T1[c]
        mask_high = channel >= T2[c]
        stretched_image[:, :, c] = np.where(
            mask_low, 0,
            np.where(mask_high, 1,
                     np.clip((channel - T1[c]) / (T2[c] - T1[c]), 0, 1))
        )
    
    # 3 色彩校正
    WB_image = stretched_image.copy()
    mean_RGB = np.mean(stretched_image)
    for c in range(3):  # 遍历 R, G, B 三个通道
        lambda_WB = np.mean(stretched_image[:, :, c]) / mean_RGB            # 三通道均值是否需要更新？？？此处不更新
        WB_image[:, :, c] = np.clip(WB_image[:, :, c] ** lambda_WB, 0, 1)

    return WB_image