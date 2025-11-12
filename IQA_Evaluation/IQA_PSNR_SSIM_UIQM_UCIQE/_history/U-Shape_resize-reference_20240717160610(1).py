# Usage:
# eg. python U-Shape_resize-reference.py .\Test-110 .\UIEB-reference-890


import sys
import os
import cv2



def main():
    # 想要resize的图片路径，取文件名
    name_path = sys.argv[1]
    name_dirs = os.listdir(name_path)
    # 需要resize的参考图像路径
    reference_path = sys.argv[2]

    N=0
    imagenum=1 # 目前处理图片是第几张
    for imgdir in name_dirs:
        if '.png' in imgdir or '.jpg' in imgdir : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('.png_out.png'):  # 文件夹内固定命名格式的图片
        # if  imgdir.endswith('_out.png'): # 文件后缀   随机应变改！！！
            reference0 = cv2.imread(os.path.join(reference_path,imgdir))
            # U-Shape
            reference = cv2.resize(reference0,(256,256))
            # U-Shape
            #第几张：图片名称
            imgname = imgdir.split('.')[0] # 以"."分为前后两部分 随机应变改！！！
            print(imagenum,":",imgname)
            imagenum += 1
            
            # 保存裁剪文件至文件夹
            ex='reference_256' # 文件夹名称
            res_folder = os.path.join(reference_path, ex)
            if not os.path.exists(res_folder):
                os.makedirs(res_folder)
            # 拼接保存图片的路径
            res_img_path = os.path.join(reference_path, ex, imgdir)
            
            # 检查同名文件是否已存在，如果存在，则删除它
            if os.path.exists(res_img_path):
                os.remove(res_img_path)
                
            # 保存
            cv2.imwrite(res_img_path, reference)

    

if __name__ == '__main__':
    main()
