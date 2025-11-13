# https://github.com/CXH-Research/GuidedHybSensUIR?tab=readme-ov-file
import json
import warnings
import torchvision
import time
import os
os.environ['TORCH_HOME'] = './weights_retrain'
from metrics.uciqe import batch_uciqe
from metrics.uiqm import batch_uiqm

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import Config
from data import get_data, get_data_nonref
from models import *
from ptflops import get_model_complexity_info

from utils import *

warnings.filterwarnings('ignore')
   
# ========== 模型参数和 FLOPs ==========
# 定义一个包装类，使得 ptflops 只传一个输入
class WrappedRCGNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)  # 用同一个 x 作为4个输入的占位符


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    # Data Loader
    val_dir = opt.TESTING.VAL_DIR

    # 有参考读取
    # val_dataset = get_data(val_dir, opt.TESTING.INPUT, opt.TESTING.TARGET, 'test', opt.TRAINING.ORI,
    #                        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})

    # # 无参考读取 resize 256
    # val_dataset = get_data_nonref(val_dir, opt.TESTING.INPUT, 'test', opt.TRAINING.ORI,
    #                        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})

    # original size
    val_dataset = get_data_nonref(val_dir, opt.TESTING.INPUT, 'test', True,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})

    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Metrics
    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    wrapped_model = WrappedRCGNet(model)
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


    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_uciqe = 0
    stat_uiqm = 0
    num = 0
    total_time = 0.0
    start_time = time.time()

    for _, test_data in enumerate(tqdm(testloader)):
        num += 1
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        # tar = test_data[1]

        with torch.no_grad():
            res = model(inp)

        if not os.path.isdir(opt.TESTING.RESULT_DIR):
            os.makedirs(opt.TESTING.RESULT_DIR)
        torchvision.utils.save_image(res, os.path.join(opt.TESTING.RESULT_DIR, test_data[1][0]))

        # stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        # stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        # stat_lpips += criterion_lpips(res, tar).item()
        stat_uciqe += batch_uciqe(res)
        stat_uiqm += batch_uiqm(res)

    # stat_psnr /= size
    # stat_ssim /= size
    # stat_lpips /= size
    stat_uciqe /= size
    stat_uiqm /= size

    end_time = time.time()
    total_time += end_time - start_time
    print("Total time: ", total_time, ", inference time: ", total_time/num)

    test_info = ("Test Result on {}, check point {}, testing data {}".
                 format(opt.MODEL.SESSION, opt.TESTING.WEIGHT, opt.TESTING.VAL_DIR))
    log_stats = ("PSNR: {}, SSIM: {}, LPIPS: {}, UCIQUE: {}, UIQM: {}".
                 format(stat_psnr, stat_ssim, stat_lpips, stat_uciqe, stat_uiqm))
    print(test_info)
    print(log_stats)
    # with open(os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode='a', encoding='utf-8') as f:
    #     f.write(json.dumps(test_info) + '\n')
    #     f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    test()
