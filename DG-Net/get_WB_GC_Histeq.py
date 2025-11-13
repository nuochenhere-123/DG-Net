
import os
import glob
import matplotlib.pyplot as plt
import shutil
from PIL import Image  # for loading images as YCbCr format
import numpy as np
import os
import skimage.io as io
import cv2
import time

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
        q = [satLevel1[ch], 1 - satLevel2[ch]]
        tiles = np.quantile(im_rgb_flat[ch, :], q)
        temp = im_rgb_flat[ch, :]
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        wb[ch, :] = temp
        bottom = min(wb[ch, :])
        top = max(wb[ch, :])
        wb[ch, :] = (wb[ch, :] - bottom) * 255 / (top - bottom)

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



def prepare_data(dataset, isRawImages=True):
  # import pdb 
  # pdb.set_trace()
  
  # 收集所有符合条件的文件（jpg、png、jpeg）
  data_dir = os.path.join(os.getcwd(), dataset)
  print("datadir_____:",data_dir)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data += glob.glob(os.path.join(data_dir, "*.jpg"))
  data += glob.glob(os.path.join(data_dir, "*.jpeg"))
  print("data:_____",len(data))
  # 退化图像需要计算 伽马校正 与 直方图均衡化 结果
  if isRawImages:
    # 更新 白平衡、伽马校正 和 直方图均衡化 处理后的图像保存目录
    wb_dir = os.path.join(os.getcwd(), f"{dataset}-wb")
    if os.path.exists(wb_dir):
      shutil.rmtree(wb_dir)  # 删除目录及其所有内容
    os.makedirs(wb_dir)  # 重新创建目录
    # gc_dir = os.path.join(os.getcwd(), f"{dataset}-gc")
    # if os.path.exists(gc_dir):
    #   shutil.rmtree(gc_dir)  # 删除目录及其所有内容
    # os.makedirs(gc_dir)  # 重新创建目录
    # histeq_dir = os.path.join(os.getcwd(), f"{dataset}-histeq")
    # if os.path.exists(histeq_dir):
    #   shutil.rmtree(histeq_dir)  # 删除目录及其所有内容
    # os.makedirs(histeq_dir)  # 重新创建目录
    num = 0
    # 对图像进行处理并保存
    total_time = 0.0
    start_time = time.time()
    for file in data:
      num = num + 1
      print("开始处理第 ",num," 张图像的 WB & GC & Histeq 变换")
      # 读取图像
      img = get_image_original(file, is_grayscale=False)
      # 应用白平衡
      img_wb = get_images_wb(img)
      # img_wb = apply_white_balance(img)
      img_wb = np.array(img_wb)
      wb_filename = os.path.join(wb_dir, os.path.basename(file))
      plt.imsave(wb_filename, img_wb)
      # # 应用伽马校正
      # img_gc = get_images_gc(img, gamma=0.7)
      # img_gc = np.array(img_gc)
      # gc_filename = os.path.join(gc_dir, os.path.basename(file))
      # plt.imsave(gc_filename, img_gc)
      # # 应用直方图均衡化
      # img_histeq = get_images_histeq(img)
      # img_histeq = np.array(img_histeq)
      # histeq_filename = os.path.join(histeq_dir, os.path.basename(file))
      # plt.imsave(histeq_filename, img_histeq)

    # 白平衡 收集所有符合条件的文件（jpg、png、jpeg）
    data_wb_dir = os.path.join(os.getcwd(), f"{dataset}-wb")
    print("data_wb_dir________:",data_wb_dir)
    wb_data = glob.glob(os.path.join(data_wb_dir, "*.png"))
    wb_data += glob.glob(os.path.join(data_wb_dir, "*.jpg"))
    wb_data += glob.glob(os.path.join(data_wb_dir, "*.jpeg"))
    print("wb_data:_____",len(wb_data))
    # # 伽马校正 收集所有符合条件的文件（jpg、png、jpeg）
    # data_gc_dir = os.path.join(os.getcwd(), f"{dataset}-gc")
    # gc_data = glob.glob(os.path.join(data_gc_dir, "*.png"))
    # gc_data += glob.glob(os.path.join(data_gc_dir, "*.jpg"))
    # gc_data += glob.glob(os.path.join(data_gc_dir, "*.jpeg"))
    # print("gc_data:_____",len(gc_data))
    # # 直方图均衡化 收集所有符合条件的文件（jpg、png、jpeg）
    # data_histeq_dir = os.path.join(os.getcwd(), f"{dataset}-histeq")
    # histeq_data = glob.glob(os.path.join(data_histeq_dir, "*.png"))
    # histeq_data += glob.glob(os.path.join(data_histeq_dir, "*.jpg"))
    # histeq_data += glob.glob(os.path.join(data_histeq_dir, "*.jpeg"))
    # print("histeq_data:_____",len(histeq_data))

    # 使用完整路径进行排序（考虑路径和扩展名）
    sorted_data = sorted(data, key=lambda x: os.path.abspath(x))
    sorted_wb_data = sorted(wb_data, key=lambda x: os.path.abspath(x))
    # sorted_gc_data = sorted(gc_data, key=lambda x: os.path.abspath(x))
    # sorted_histeq_data = sorted(histeq_data, key=lambda x: os.path.abspath(x))  
    
    end_time = time.time()
    total_time = end_time - start_time
    print("total time: ", total_time)

    # return sorted_data, sorted_wb_data, sorted_gc_data, sorted_histeq_data
    return sorted_data, sorted_wb_data

  else: # 是参考数据
    # 使用完整路径进行排序（考虑路径和扩展名）
    sorted_data = sorted(data, key=lambda x: os.path.abspath(x))
    
    return sorted_data
  


# data_train_list, data_train_list_wb, data_train_list_gc, data_train_list_histeq   = prepare_data(dataset="./datasets/780/try", isRawImages=True) # UIEB-860 + UFO-40
data_train_list, data_train_list_wb  = prepare_data(dataset="./datasets/780/raw-780", isRawImages=True) # UIEB-860 + UFO-40
