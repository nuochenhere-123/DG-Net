
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t
from data.addNoise import getRandomNoise
from data.getWbGcHis import *
from torchvision.transforms import ToTensor

    
class UIEDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(UIEDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        # self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/raw-780'
        folder_wb = self.data_dir+'/raw-780-wb'
        folder_gc = self.data_dir+'/raw-780-gc'
        folder_his = self.data_dir+'/raw-780-histeq'
        folder2= self.data_dir+'/reference-780'
        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        data_filenames_wb = sorted([join(folder_wb, x) for x in listdir(folder_wb) if is_image_file(x)])
        data_filenames_gc = sorted([join(folder_gc, x) for x in listdir(folder_gc) if is_image_file(x)])
        data_filenames_his = sorted([join(folder_his, x) for x in listdir(folder_his) if is_image_file(x)])
        data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im1_wb = load_img(data_filenames_wb[index])
        im1_gc = load_img(data_filenames_gc[index])
        im1_his = load_img(data_filenames_his[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file1_wb = os.path.split(data_filenames_wb[index])
        _, file1_gc = os.path.split(data_filenames_gc[index])
        _, file1_his = os.path.split(data_filenames_his[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)

            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_wb = self.transform(im1_wb)
            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_gc = self.transform(im1_gc)
            random.seed(seed)
            torch.manual_seed(seed) 
            im1_his = self.transform(im1_his)

            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        
        return im1, im1_wb, im1_gc, im1_his, im2, file1, file1_wb, file1_gc, file1_his, file2

    def __len__(self):
        return 780


class UIEDatasetFromFolder_with_NR(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(UIEDatasetFromFolder_with_NR, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        # self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/raw-780'
        folder_wb = self.data_dir+'/raw-780-wb'
        folder_gc = self.data_dir+'/raw-780-gc'
        folder_his = self.data_dir+'/raw-780-histeq'
        folder2= self.data_dir+'/reference-780'
        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        data_filenames_wb = sorted([join(folder_wb, x) for x in listdir(folder_wb) if is_image_file(x)])
        data_filenames_gc = sorted([join(folder_gc, x) for x in listdir(folder_gc) if is_image_file(x)])
        data_filenames_his = sorted([join(folder_his, x) for x in listdir(folder_his) if is_image_file(x)])
        data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        # NR
        folder_NR = self.data_dir+'/LSUI-780'
        data_filenames_NR = sorted([join(folder_NR, x) for x in listdir(folder_NR) if is_image_file(x)])

        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im1_wb = load_img(data_filenames_wb[index])
        im1_gc = load_img(data_filenames_gc[index])
        im1_his = load_img(data_filenames_his[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file1_wb = os.path.split(data_filenames_wb[index])
        _, file1_gc = os.path.split(data_filenames_gc[index])
        _, file1_his = os.path.split(data_filenames_his[index])
        _, file2 = os.path.split(data_filenames2[index])
        # NR
        im_NR = load_img(data_filenames_NR[index])
        _, file_NR = os.path.split(data_filenames_NR[index])

        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)

            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_wb = self.transform(im1_wb)
            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_gc = self.transform(im1_gc)
            random.seed(seed)
            torch.manual_seed(seed) 
            im1_his = self.transform(im1_his)

            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 

            random.seed(seed)
            torch.manual_seed(seed)         
            im_NR = self.transform(im_NR) 

        # im_NR 为 tensor (0-1, CxHxW)    需要 → NumPy (0-255, HxWxC)
        img_np = im_NR.permute(1, 2, 0).numpy()  # 0-1 float32
        
        im_NR_wb = get_images_wb(np.array(img_np)) # get_images_wb 要求输入为0-1，自动输出0-1
        im_NR_gc = get_images_gc(np.array(img_np))
        im_NR_his = get_images_histeq(np.array(img_np))
        # im_NR_cp_his = get_his_after_cp(np.array(img_np))

        # unit8 转 array 转 tensor，ToTensor() 会把输入的 uint8 图像（0–255）自动归一化为 0–1 的 float32 tensor
        im_NR_wb   = ToTensor()(Image.fromarray((im_NR_wb * 255).astype(np.uint8)))
        im_NR_gc   = ToTensor()(Image.fromarray((im_NR_gc * 255).astype(np.uint8)))
        im_NR_his  = ToTensor()(Image.fromarray((im_NR_his * 255).astype(np.uint8)))
        # im_NR_cp_his  = ToTensor()(Image.fromarray((im_NR_cp_his * 255).astype(np.uint8)))
        
        return im1, im1_wb, im1_gc, im1_his, im2, file1, file1_wb, file1_gc, file1_his, file2, im_NR, im_NR_wb, im_NR_gc, im_NR_his

    def __len__(self):
        return 780


class UIEDatasetFromFolder_withNoise(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(UIEDatasetFromFolder_withNoise, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        # self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/raw-780'
        folder_wb = self.data_dir+'/raw-780-wb'
        folder_gc = self.data_dir+'/raw-780-gc'
        folder_his = self.data_dir+'/raw-780-histeq'
        folder2= self.data_dir+'/reference-780'
        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        data_filenames_wb = sorted([join(folder_wb, x) for x in listdir(folder_wb) if is_image_file(x)])
        data_filenames_gc = sorted([join(folder_gc, x) for x in listdir(folder_gc) if is_image_file(x)])
        data_filenames_his = sorted([join(folder_his, x) for x in listdir(folder_his) if is_image_file(x)])
        data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im1_wb = load_img(data_filenames_wb[index])
        im1_gc = load_img(data_filenames_gc[index])
        im1_his = load_img(data_filenames_his[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file1_wb = os.path.split(data_filenames_wb[index])
        _, file1_gc = os.path.split(data_filenames_gc[index])
        _, file1_his = os.path.split(data_filenames_his[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)

            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_wb = self.transform(im1_wb)
            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_gc = self.transform(im1_gc)
            random.seed(seed)
            torch.manual_seed(seed) 
            im1_his = self.transform(im1_his)

            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 

        # im1 为 tensor (0-1, CxHxW)    需要 → NumPy (0-255, HxWxC)
        img_np = im1.permute(1, 2, 0).numpy() * 255.0  # 0-255 float32

        # 加随机噪声，输入np数组，返回 PIL 格式（0–255）
        im1_noise = ( getRandomNoise(img_np).astype(np.float32) )/255.0  # getRandomNoise 输出 0–255, uint8, im1_noise 为 0-1 float32

        im1_noise_wb = get_images_wb(np.array(im1_noise)) # get_images_wb 要求输入为0-1，自动输出0-1
        im1_noise_gc = get_images_gc(np.array(im1_noise))
        im1_noise_his = get_images_histeq(np.array(im1_noise))

        # unit8 转 array 转 tensor，ToTensor() 会把输入的 uint8 图像（0–255）自动归一化为 0–1 的 float32 tensor
        im1_noise      = ToTensor()(Image.fromarray((im1_noise * 255).astype(np.uint8)))
        im1_noise_wb   = ToTensor()(Image.fromarray((im1_noise_wb * 255).astype(np.uint8)))
        im1_noise_gc   = ToTensor()(Image.fromarray((im1_noise_gc * 255).astype(np.uint8)))
        im1_noise_his  = ToTensor()(Image.fromarray((im1_noise_his * 255).astype(np.uint8)))
        
        # 要求转 tensor 返回, 0-1
        return im1, im1_wb, im1_gc, im1_his, im2, file1, file1_wb, file1_gc, file1_his, file2, im1_noise, im1_noise_wb, im1_noise_gc, im1_noise_his

    def __len__(self):
        return 780


class UIEDatasetFromFolder_withNoise_NR(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(UIEDatasetFromFolder_withNoise_NR, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):

        folder = self.data_dir+'/raw-780'
        folder_wb = self.data_dir+'/raw-780-wb'
        folder_gc = self.data_dir+'/raw-780-gc'
        folder_his = self.data_dir+'/raw-780-histeq'
        folder2= self.data_dir+'/reference-780'
        folder_NR= self.data_dir+'/780_NR_NUID'
        folder_NR_wb= self.data_dir+'/780_NR_NUID-wb'
        folder_NR_gc= self.data_dir+'/780_NR_NUID-gc'
        folder_NR_his= self.data_dir+'/780_NR_NUID-histeq'
        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        data_filenames_wb = sorted([join(folder_wb, x) for x in listdir(folder_wb) if is_image_file(x)])
        data_filenames_gc = sorted([join(folder_gc, x) for x in listdir(folder_gc) if is_image_file(x)])
        data_filenames_his = sorted([join(folder_his, x) for x in listdir(folder_his) if is_image_file(x)])
        data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])
        data_filenames_NR = sorted([join(folder_NR, x) for x in listdir(folder_NR) if is_image_file(x)])
        data_filenames_NR_wb = sorted([join(folder_NR, x) for x in listdir(folder_NR_wb) if is_image_file(x)])
        data_filenames_NR_gc = sorted([join(folder_NR, x) for x in listdir(folder_NR_gc) if is_image_file(x)])
        data_filenames_NR_his = sorted([join(folder_NR, x) for x in listdir(folder_NR_his) if is_image_file(x)])
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im1_wb = load_img(data_filenames_wb[index])
        im1_gc = load_img(data_filenames_gc[index])
        im1_his = load_img(data_filenames_his[index])
        im2 = load_img(data_filenames2[index])
        im_NR = load_img(data_filenames_NR[index])
        im_NR_wb = load_img(data_filenames_NR_wb[index])
        im_NR_gc = load_img(data_filenames_NR_gc[index])
        im_NR_his = load_img(data_filenames_NR_his[index])
        # _, file1 = os.path.split(data_filenames[index])
        # _, file1_wb = os.path.split(data_filenames_wb[index])
        # _, file1_gc = os.path.split(data_filenames_gc[index])
        # _, file1_his = os.path.split(data_filenames_his[index])
        # _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_wb = self.transform(im1_wb)
            random.seed(seed) 
            torch.manual_seed(seed) 
            im1_gc = self.transform(im1_gc)
            random.seed(seed)
            torch.manual_seed(seed) 
            im1_his = self.transform(im1_his)

            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 

            random.seed(seed)
            torch.manual_seed(seed)         
            im_NR = self.transform(im_NR) 
            random.seed(seed)
            torch.manual_seed(seed)         
            im_NR_wb = self.transform(im_NR_wb) 
            random.seed(seed)
            torch.manual_seed(seed)         
            im_NR_gc = self.transform(im_NR_gc) 
            random.seed(seed)
            torch.manual_seed(seed)         
            im_NR_his = self.transform(im_NR_his) 

        # im_NR 为 tensor (0-1, CxHxW)    需要 → NumPy (0-255, HxWxC)
        img_np = im_NR.permute(1, 2, 0).numpy() * 255.0  # 0-255 float32

        # 加随机噪声，输入np数组，返回 PIL 格式（0–255）
        im_NR_noise = ( getRandomNoise(img_np).astype(np.float32) )/255.0  # getRandomNoise 输出 0–255, uint8, im1_noise 为 0-1 float32

        im_NR_noise_wb = get_images_wb(np.array(im_NR_noise)) # get_images_wb 要求输入为0-1，自动输出0-1
        im_NR_noise_gc = get_images_gc(np.array(im_NR_noise))
        im_NR_noise_his = get_images_histeq(np.array(im_NR_noise))

        # unit8 转 array 转 tensor，ToTensor() 会把输入的 uint8 图像（0–255）自动归一化为 0–1 的 float32 tensor
        im_NR_noise      = ToTensor()(Image.fromarray((im_NR_noise * 255).astype(np.uint8)))
        im_NR_noise_wb   = ToTensor()(Image.fromarray((im_NR_noise_wb * 255).astype(np.uint8)))
        im_NR_noise_gc   = ToTensor()(Image.fromarray((im_NR_noise_gc * 255).astype(np.uint8)))
        im_NR_noise_his  = ToTensor()(Image.fromarray((im_NR_noise_his * 255).astype(np.uint8)))
        
        # 要求转 tensor 返回, 0-1
        return im1, im1_wb, im1_gc, im1_his, im2, im_NR, im_NR_wb, im_NR_gc, im_NR_his, im_NR_noise, im_NR_noise_wb, im_NR_noise_gc, im_NR_noise_his

    def __len__(self):
        return 780