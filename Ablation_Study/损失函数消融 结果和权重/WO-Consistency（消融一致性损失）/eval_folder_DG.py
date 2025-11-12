from net.DG_Net import DGNet
from data.getWbGcHis import *
import os
import json
import safetensors.torch as sf
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import platform
import time
from PIL import Image
from ptflops import get_model_complexity_info
from pathlib import Path

eval_parser = argparse.ArgumentParser(description='EvalHF')
eval_parser.add_argument('--path', type=str, default="./weights_DG/train_DG_WO-Consistency/epoch_150.pth", help='You can change this path to our method weights mentioned here: https://huggingface.co/papers/2502.20272.')
eval_parser.add_argument('--input_folder', type=str, default="./Test/Test-LSUI-350", help='The path of your image.')
# eval_parser.add_argument('--input_folder', type=str, default="./Test/noise-checker/checker-rayleigh-5", help='The path of your image.')
# eval_parser.add_argument('--output_folder', type=str, default="./results_Cycle_FR&NR/results_squareLimit_EveryLimit_0.02color+0.004color_var+all_raw/epoch-200/res-NUID", help='The path of your output.')
eval_parser.add_argument('--output_folder', type=str, default="./results_DG/results_WO-Consistency/res-LSUI-350", help='The path of your output.')
el = eval_parser.parse_args()

def from_pretrained(cls, pretrained_model_name_or_path: str):
    # 如果路径是本地路径，则直接加载
    if os.path.exists(pretrained_model_name_or_path):
        print(f"Loading weights from local file: {pretrained_model_name_or_path}")
        state_dict = torch.load(pretrained_model_name_or_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        cls.load_state_dict(state_dict, strict=False)
        return cls


# ========== 初始化模型 ==========
model = DGNet().cuda()
model = from_pretrained(cls=model,pretrained_model_name_or_path=el.path)
model.eval()

     
# 推理前（计算最大显存占用情况）    
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device=0)

# ========== 模型参数和 FLOPs ==========
# 定义一个包装类，使得 ptflops 只传一个输入
class WrappedRCGNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x, x, x, x)  # 用同一个 x 作为4个输入的占位符

wrapped_model = WrappedRCGNet(model)
with torch.cuda.device(0):
    gmacs, params = get_model_complexity_info(
        wrapped_model,
        (3, 256, 256),
        as_strings=True,
        print_per_layer_stat=False   # 打印信息
    )
# 将 GMACs 转换为 FLOPs 和 GFLOPS
gflops = float(gmacs.split(' ')[0]) * 2   # 将 GMACs 转换为 FLOPs 再转为 GFLOPS
print(f"Total Params: {params}")
print(f"Total FLOPs: {gflops} GFLOPS\n")


# 创建输出文件夹
output_folder = el.output_folder
if not os.path.exists(output_folder):          
    os.makedirs(output_folder)

# # 构建光照和反射率保存的子目录路径
# illumination_dir = os.path.join(output_folder, "illumination")
# os.makedirs(illumination_dir, exist_ok=True)
 

# 图像转换 将图像从 HWC 格式（高度、宽度、通道数）转换为 CHW 格式（通道数、高度、宽度），并将图像像素值归一化到 [0, 1] 范围
pil2tensor = transforms.Compose([transforms.ToTensor()])

# 处理整个文件夹中的所有图像
input_folder = el.input_folder
image_extensions = ('.png', '.jpg', '.jpeg')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
sum = 0
total_time = 0.0
start_time = time.time()
for image_name in image_files:
    sum += 1
    image_path = os.path.join(input_folder, image_name)
    print(f"Processing {sum}: {image_path}...")

    img = get_image_original(image_path, is_grayscale=False) # RGB格式

    # # resize 256 版本
    # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
    img_wb = get_images_wb(img)
    # img_wb = apply_white_balance(img) # 每张256图片提升0.03s
    img_gc = get_images_gc(img)
    img_his = get_images_histeq(img)
    
    input_tensor = pil2tensor(img)
    input_tensor_wb = pil2tensor(img_wb)
    input_tensor_gc = pil2tensor(img_gc)
    input_tensor_his = pil2tensor(img_his)

    # 确保尺寸是 8 的倍数
    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = F.pad(input_tensor.unsqueeze(0), (0,padw,0,padh), 'reflect') # (0, padw, 0, padh) 的顺序是 (左, 右, 上, 下)，所以这里只在右边和下边补像素
    input_tensor_wb = F.pad(input_tensor_wb.unsqueeze(0), (0,padw,0,padh), 'reflect')
    input_tensor_gc = F.pad(input_tensor_gc.unsqueeze(0), (0,padw,0,padh), 'reflect')
    input_tensor_his = F.pad(input_tensor_his.unsqueeze(0), (0,padw,0,padh), 'reflect')
    
    # 确保最终的 tensor 还是 float32 类型
    input_tensor = input_tensor.float()
    input_tensor_wb = input_tensor_wb.float()
    input_tensor_gc = input_tensor_gc.float()
    input_tensor_his = input_tensor_his.float()

    torch.cuda.synchronize()
    with torch.no_grad():
        # model.trans.alpha_s = el.alpha_s
        # model.trans.alpha = el.alpha_i
        output_tensor, L, output_raw = model(input_tensor.cuda(),input_tensor_wb.cuda(),input_tensor_gc.cuda(),input_tensor_his.cuda())
        # output_tensor, L, output_raw, reflection_out, noise_out = model(input_tensor.cuda(),input_tensor_wb.cuda(),input_tensor_gc.cuda(),input_tensor_his.cuda())
    torch.cuda.synchronize()

    # 裁剪回原始尺寸
    output_tensor = torch.clamp(output_tensor.cuda(),0,1).cuda()[:, :, :h, :w]
    output_raw = torch.clamp(output_raw.cuda(), 0, 1).cuda()[:, :, :h, :w]
    
    # 转换为 PIL 并保存
    enhanced_img = transforms.ToPILImage()(output_tensor.squeeze(0))  # 去掉 batch 维度
    enhanced_img.save(os.path.join(output_folder, image_name))

    # 保存 output_raw
    raw_folder = os.path.join(output_folder, "raw")
    os.makedirs(raw_folder, exist_ok=True)  # 如果不存在就创建
    # 构造 raw 图像名称
    # raw_name = os.path.splitext(image_name)[0] + "_raw" + os.path.splitext(image_name)[1]
    raw_name = image_name
    # raw 图像保存到 raw 文件夹中
    raw_img = transforms.ToPILImage()(output_raw.squeeze(0))
    raw_img.save(os.path.join(raw_folder, raw_name))

    # # 保存光照
    # L = torch.clamp(L.cuda(), 0, 1).cuda()[:, :, :h, :w]
    # # L = L[:, :, :h, :w]  # 同样裁剪 L
    # # 保存 L 三通道整体图像 和 反射率 R
    # L_full_img = transforms.ToPILImage()(L.squeeze(0))  # [3,H,W] 转 PIL
    # L_full_img.save(os.path.join(output_folder, os.path.splitext(image_name)[0] + '_illumination.png'))
   
    # # 保存 L 的三个通道分别为 R/G/B
    # L_channels = torch.chunk(L.squeeze(0), 3, dim=0)  # 分割为3个 [1,H,W] tensor
    # channel_names = ['R', 'G', 'B']
    # for ch, name in zip(L_channels, channel_names):
    #     ch_img = transforms.ToPILImage()(ch)  # 单通道转换
    #     ch_img.save(os.path.join(illumination_dir, os.path.splitext(image_name)[0] + f'_{name}.png'))

    # # 保存噪声
    # noise_out = torch.clamp(noise_out.cuda(), 0, 1).cuda()[:, :, :h, :w]
    # # 保存 L 三通道整体图像 和 反射率 R
    # Noise_full_img = transforms.ToPILImage()(noise_out.squeeze(0))  # [3,H,W] 转 PIL
    # Noise_full_img.save(os.path.join(output_folder, os.path.splitext(image_name)[0] + '_noise.png'))

    #  # 保存R_sum
    # reflection_out = torch.clamp(reflection_out.cuda(), 0, 1).cuda()[:, :, :h, :w]
    # # 保存 L 三通道整体图像 和 反射率 R
    # Rsum_full_img = transforms.ToPILImage()(reflection_out.squeeze(0))  # [3,H,W] 转 PIL
    # Rsum_full_img.save(os.path.join(output_folder, os.path.splitext(image_name)[0] + '_Rsum.png'))
   

end_time = time.time()
total_time += end_time - start_time
print("Processing complete. Enhanced images saved in:", output_folder, "total ", sum, " images.")
print("Total time: ", total_time, ", inference time: ", total_time/sum)
# 推理后（计算最大显存占用情况）
max_mem = torch.cuda.max_memory_reserved(device=0) / (1024 ** 3)  
# 返回值是 KB为 /1024，MB为 /(1024 ** 2), GB为 /(1024 ** 3)
print(f"最大 GPU 显存占用：{max_mem:.2f} GB")

# 构造日志内容
log_lines = [
    f"Total Params: {params}", 
    f"Total FLOPs: {gflops} GFLOPS"
    f"Saved output to {output_folder}!",
    f"Total time: {total_time:.2f} s",
    f"Inference time per image: {total_time/sum:.4f} s",
    f"最大 GPU 显存占用：{max_mem:.2f} GB"
]
# 写入 log.txt
output_folder = Path(output_folder)
log_path = output_folder / "log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
