import os
import argparse
from tqdm import tqdm
import time
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from thop import profile

from data_RGB import get_test_data
from Networks.model import Net
from skimage import img_as_ubyte
from ptflops import get_model_complexity_info

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='model_C', type=str, help='Network Type')
parser.add_argument('--input_dir', default='./Test/', type=str, help='Directory of validation images') # 测试文件夹所在文件夹
parser.add_argument('--result_dir', default='./results_retrain/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints_retrain/model_C/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Test-checker', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--resdataset', default='res-checker-256', type=str, help='Test Dataset') # 目前只能8倍数输入输出，修改尺寸在 dataset_RGB.py Line 161
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = Net()
model_restoration.cuda()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ",args.weights)

# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# ========== 模型参数和 FLOPs ==========
# 定义一个包装类，使得 ptflops 只传一个输入
class WrappedRCGNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)  # 用同一个 x 作为4个输入的占位符

wrapped_model = WrappedRCGNet(model_restoration)
with torch.cuda.device(0):
    gmacs, params = get_model_complexity_info(
            wrapped_model,
            (3, 256, 256),
            as_strings=True,
            print_per_layer_stat=False
    )
# 将 GMACs 转换为 FLOPs 和 GFLOPS
gflops = float(gmacs.split(' ')[0]) * 2   # 将 GMACs 转换为 FLOPs 再转为 GFLOPS
print(f"Total Params: {params}")
print(f"Total FLOPs: {gflops} GFLOPS\n")


dataset = args.dataset
resdataset = args.resdataset
rgb_dir_test = os.path.join(args.input_dir, dataset)
print(rgb_dir_test)
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir = os.path.join(args.result_dir, resdataset)
utils.mkdir(result_dir)
num = 0
total_time = 0.0
start_time = time.time()
with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        num += 1
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored0 = torch.clamp(restored[2], 0, 1)
        restored0 = restored0.permute(0, 2, 3, 1).cpu().detach().numpy()

        # restored2 = torch.clamp(restored[1], 0, 1)
        # restored2 = restored2.permute(0, 2, 3, 1).cpu().detach().numpy()
        #
        # restored4 = torch.clamp(restored[0], 0, 1)
        # restored4 = restored4.permute(0, 2, 3, 1).cpu().detach().numpy()

        restored_img0 = img_as_ubyte(restored0[0])
        utils.save_img((os.path.join(result_dir, filenames[0]+'.png')), restored_img0)
        # restored_img2 = img_as_ubyte(restored2[0])
        # utils.save_img((os.path.join(result_dir, filenames[0]+ '2.png')), restored_img2)
        # restored_img4 = img_as_ubyte(restored4[0])
        # utils.save_img((os.path.join(result_dir, filenames[0]+ '4.png')), restored_img4)

    end_time = time.time()
    total_time += end_time - start_time
    print("Total time: ", total_time, ", inference time: ", total_time/num)

    flops, params = profile(model_restoration, inputs=(input_,))
    print('flops: ', flops, 'params: ', params)

