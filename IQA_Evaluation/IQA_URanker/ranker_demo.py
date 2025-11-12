# run: python ranker_demo.py 
from tkinter import E
import utils
import torch
import argparse
import os
import skimage.io as io
from PIL import Image
from torchvision import transforms

# # 设置 CUDA_VISIBLE_DEVICES 环境变量，指定使用 GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str, default = "options/URanker.yaml")
parser.add_argument('--checkpoint_path', type=str, default = "checkpoints/URanker_ckpt.pth")
parser.add_argument('--input_path', type=str, default = "./ToTest/RetinexBased-110/")  # 测试图像所在文件夹
# parser.add_argument('--input_path', type=str, default = "./ToTest/UShape-110-resize/")
parser.add_argument('--save_path', type=str, default = "./results/RetinexBased-110-256-URanker.txt") # 指标结果保存文件夹
# parser.add_argument('--save_path', type=str, default = "./results/UShape-110-resize-URanker.txt")
args = parser.parse_args()

options = utils.get_option(args.opt_path)
options['model']['resume_ckpt_path'] = args.checkpoint_path
model = utils.build_model(options['model'])
filenames = os.listdir(args.input_path)
sum_ranker = 0
num = 0
with open(args.save_path, 'w') as f:
    f.write(f'{args.input_path}: \n')
    for filename in filenames:
        if '.png' in filename or '.jpg' in filename : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('io.png'):  # 文件夹内固定命名格式的图片
        # if  imgdir.endswith('original.png'): # 文件后缀
        # if  filename.endswith('_out.png') or filename.endswith('_out.jpg') or filename.endswith('_out.jpeg'): # 文件后缀
            filepath = os.path.join(args.input_path, filename)
            img = Image.open(filepath)
            img_w, img_h = img.size[0], img.size[1]
                
            img = transforms.Resize((img_h//2, img_w//2))(img)
            img = transforms.ToTensor()(img).cuda().unsqueeze(0)
            inputs = utils.preprocessing(img)
            pred = model(**inputs)['final_result'][0][0][0]
            sum_ranker += pred.item()
            num += 1
            print(f'{num}: {filename}\t{pred}')
            f.write(f'{num}: {filename}\t{pred}\n')
            
    avg = sum_ranker/num
    print(f'\nAVG URanker: {avg}')
    f.write(f'\nAVG URanker: {avg}')
