# 将 文件夹内图片 resize 成 目标大小，如原始图像与reference大小相同，但模型输入原始大小的图像后输出为256*256
#   此处 可将任意大小（如 256*256）自动 resize 回原始大小

# Usage: 以下参数修改后 直接 run

# 1. 修改 a. 想要resize的图片路径 resize_path  12行
#         b. resize大小的参考图像路径 reference_path  15行
#         c. resize 后保存路径 save_dir  17行
# 2. 修改需要更改尺寸的图片命名格式  28行
# 3. 提取原始图像前缀，寻找参考图像 38行

import cv2
import os
import numpy as np
from PIL import Image 

if __name__ == '__main__':
    # 想要resize的图片路径
    resize_path = "./FUnIE-C60"
    resize_dirs = os.listdir(resize_path)
    # resize大小的参考图像路径
    # reference_path = "./UIEB-reference-890"
    reference_path = "./Size_Reference/Test-C60"
    # resize 后保存路径
    save_dir = "./FUnIE-C60-resize"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    N=0
    imagenum=1 # 目前处理图片是第几张
    for imgdir in resize_dirs:
        if '.png' in imgdir or '.jpg' or '.jpeg' in imgdir : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('io.png'):  # 文件夹内固定命名格式的图片
        # if  imgdir.endswith('original.png'): # 文件后缀
        # if  imgdir.endswith('_out.png') or imgdir.endswith('_out.jpg') or imgdir.endswith('_out.jpeg'): # 文件后缀
            
            # 读取需要更改Size的图片（io读则io保存RGB，cv2读则cv2保存BGR）
            image_to_resize = cv2.imread(os.path.join(resize_path, imgdir))
            # 图片名称为imgdir，此处提取前缀
            # imgname_cor = imgdir.split('_out.')[0]
            # refname = imgname_cor+'.'+imgdir.split('_out.')[1] # 参考图像名称（含后缀）随机应变改！！！
            imgname_cor = imgdir.split('.')[0]
            refname = imgname_cor+'.'+imgdir.split('.')[1] # 参考图像名称（含后缀）随机应变改！！！
            print("处理第", imagenum, "张图像：", os.path.join(resize_path,imgdir))
            
            # 寻找对应的reference
            file_path1 = os.path.join(reference_path, refname)
            ref_image = cv2.imread(file_path1)
            ref_shape = ref_image.shape
            ref_size = (ref_shape[1], ref_shape[0])
            
            # resize回 原大小 保存
            to_resize_1 = image_to_resize.astype(np.uint8)
            a=Image.fromarray(to_resize_1)
            b=a.resize(ref_size) # PIL 使用的是 双线性插值 作为默认的缩放方法
            resize_res = np.array(b)
            save_img_dir = os.path.join(save_dir, imgdir)
            # 检查同名文件是否已存在，如果存在，则删除它
            if os.path.exists(save_img_dir):
                os.remove(save_img_dir)
            cv2.imwrite(save_img_dir, resize_res) 
            imagenum += 1
            
            

    