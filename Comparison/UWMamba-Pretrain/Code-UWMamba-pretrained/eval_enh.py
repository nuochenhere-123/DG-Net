# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pprint
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import time
import timeit
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

# import _init_paths
import lib.models
import lib.datasets
from lib.config import config
from lib.config import update_config
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, speed_test

import math
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # ------------------------LSUI-----------------------------
                        # default="./experiments/LSUI/UWMamba.yaml",


                        # ------------------------UIEB-----------------------------
                        default= "./experiments/UIEB/UWMamba.yaml",

                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('lib.models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('lib.models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)
    
    # 计算参数量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Initialized model with {} trainable params ".format(params))
    params_M = round((np.ceil(params / 10000) / 100), 1)
    print('The scale of the model is {}M，or can be said as {}'.format(params_M,params))
    
    pretrained_state = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_state = {k: v for k, v in pretrained_state.items() if
                        (k in model_dict and v.shape == model_dict[k].shape)}

    for k, _ in pretrained_state.items():
        print('=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_state)
    model.load_state_dict(model_dict, strict=False)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    # 新的文件路径
    test_dir = './test/Test-NUID-110'

    # 获取该目录下的所有图像文件路径（假设是图片文件，可以根据需要更改文件扩展名）
    test_images = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith(('.jpg', '.png', '.jpeg'))]
    # # 打印测试图像列表，确认读取的路径
    # print(test_images)
    
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('lib.datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=test_images,   # 已修改 
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    PSNR_list = []
    SSIM_list = []
    sv_dir = './test/results/Test-NUID-256'
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    start = timeit.default_timer()

    with torch.no_grad():
        model.eval()  # 切换到评估模式
        for idx, batch in enumerate(testloader):
            # print(batch)
            image, _, name = batch
            print("正在处理第", idx+1, "张图像：", name)
            # image = image.clone().detach()
            image = image.cuda(non_blocking=True)
            pred = model(image)
            # if pred.shape[2] != label.shape[2] or pred.shape[3] != label.shape[3]:
            #     pred = F.interpolate(pred, size=(label.shape[2], label.shape[3]), mode='bilinear',align_corners=True)
            pred_ = torch.squeeze(pred,dim=0).permute(1,2,0).cpu().numpy()
            # label_ = torch.squeeze(label,dim=0).permute(1,2,0).cpu().numpy()

            # score_psnr = cal_psnr(pred_, label_, data_range=1)
            # score_ssim = cal_ssim(pred_, label_, channel_axis=2, data_range=1)
            # PSNR_list.append(score_psnr)
            # SSIM_list.append(score_ssim)
            pred_vis = pred_*255
            pred_vis = pred_vis[:, :, ::-1]
            save_name = os.path.join(sv_dir, name[0] + '.jpg')
            cv2.imwrite(save_name,pred_vis)
            torch.cuda.empty_cache()
            
    # PSNR_list = np.array(PSNR_list)
    # PSNR = PSNR_list.mean()
    # SSIM_list = np.array(SSIM_list)
    # SSIM = SSIM_list.mean()

    # msg = 'PSNR: {: 4.4f}, SSIM: {: 4.4f}'.format(PSNR, SSIM)
    # logging.info(msg)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int64((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
