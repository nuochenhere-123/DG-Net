import cv2
import os
import numpy as np
from PIL import Image 

if __name__ == '__main__':
    # 想要resize的图片路径
    resize_path = ".\ToChange"
    # resize 后保存路径
    save_dir1 = ".\Resized"
    
    sum = 0
    
    # 遍历 Test 文件夹下的所有子文件夹
    for data_dir, _, _ in os.walk(resize_path):
        print("正在处理", data_dir, "文件夹")
        imagenum=1 # 目前处理图片是第几张
        for imgdir in os.listdir(data_dir):
            if '.png' in imgdir or '.jpg' in imgdir or '.jpeg' in imgdir : # 文件夹内所有图片
            # if imgdir.endswith('.png') and not imgdir.endswith('io.png'):  # 文件夹内固定命名格式的图片
            # if  imgdir.endswith('original.png'): # 文件后缀
            # if  imgdir.endswith('_out.png') or imgdir.endswith('_out.jpg') or imgdir.endswith('_out.jpeg'): # 文件后缀
                
                # 保存路径
                save_dir = os.path.join(save_dir1, os.path.relpath(data_dir, resize_path))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                # 读取需要更改Size的图片（io读则io保存RGB，cv2读则cv2保存BGR）
                image_to_resize = cv2.imread(os.path.join(data_dir, imgdir))
                # 图片名称为imgdir，此处提取前缀
                # imgname_cor = imgdir.split('_out.')[0]
                # refname = imgname_cor+'.'+imgdir.split('_out.')[1] # 参考图像名称（含后缀）随机应变改！！！
                imgname_cor = imgdir.split('.')[0]
                # refname = imgname_cor+'.'+imgdir.split('.')[1] # 参考图像名称（含后缀）随机应变改！！！
                # print("处理第", imagenum, "张图像：", os.path.join(data_dir, imgdir))
                
                # resize回 原大小 保存
                to_resize_1 = image_to_resize.astype(np.uint8)
                a=Image.fromarray(to_resize_1)
                ref_size = (256, 256)
                b=a.resize(ref_size) # PIL 使用的是 双线性插值 作为默认的缩放方法
                resize_res = np.array(b)
                save_img_dir = os.path.join(save_dir, imgdir)
                # 检查同名文件是否已存在，如果存在，则删除它
                if os.path.exists(save_img_dir):
                    os.remove(save_img_dir)
                cv2.imwrite(save_img_dir, resize_res) 
                imagenum += 1
        print("已经完成对", imagenum - 1, "张图片尺寸的修改")
        sum += imagenum - 1
        
    print("\n共计 已完成 对 ", sum, " 张 图片尺寸 的 修改")