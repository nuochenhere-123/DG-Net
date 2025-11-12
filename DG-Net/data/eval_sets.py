
import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F
    
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        folder = data_dir+'/raw-60'
        folder_wb = data_dir+'/raw-60-wb'
        folder_gc = data_dir+'/raw-60-gc'
        folder_his = data_dir+'/raw-60-histeq'
        data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        data_filenames_wb = sorted([join(folder_wb, x) for x in listdir(folder_wb) if is_image_file(x)])
        data_filenames_gc = sorted([join(folder_gc, x) for x in listdir(folder_gc) if is_image_file(x)])
        data_filenames_his = sorted([join(folder_his, x) for x in listdir(folder_his) if is_image_file(x)])

        self.data_filenames = data_filenames
        self.data_filenames_wb = data_filenames_wb
        self.data_filenames_gc = data_filenames_gc
        self.data_filenames_his = data_filenames_his
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        input_wb = load_img(self.data_filenames_wb[index])
        _, file_wb = os.path.split(self.data_filenames_wb[index])
        input_gc = load_img(self.data_filenames_gc[index])
        _, file_gc = os.path.split(self.data_filenames_gc[index])
        input_his = load_img(self.data_filenames_his[index])
        _, file_his = os.path.split(self.data_filenames_his[index])

        if self.transform: # 转换到 0-1
            input = self.transform(input)
            input_wb = self.transform(input_wb)
            input_gc = self.transform(input_gc)
            input_his = self.transform(input_his)
            
        return input, input_wb, input_gc, input_his, file, file_wb, file_gc, file_his

    def __len__(self):
        return len(self.data_filenames)