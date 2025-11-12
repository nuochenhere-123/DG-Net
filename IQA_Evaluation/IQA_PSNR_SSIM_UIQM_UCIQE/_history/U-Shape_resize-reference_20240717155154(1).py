
# Metrics for underwater image quality evaluation of FR & NR.

# Usage:
#  $ python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py RESULT_PATH REFERENCE_PATH
# eg. python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\result .\real_inference
#     python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\upgrade2_3-R90 .\UIEB-reference-890
#     python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\Test-110_2_3 .\UIEB-reference-890
# 每次运行修改：1. 保存文件夹名   2. 处理图片命名格式   3. 保存txt文件名

# 保存带指标图片 至 图片路径的 'Res_PSNR_SSIM_UCIQE_inLAB_ucolor' 文件夹 中（第87行ex变量）

import numpy as np
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import sys
import os
import cv2
from skimage import color, filters
import math

# 保存带指标的图片到结果路径下文件夹内[输入图像所在文件夹，输入图像，输入图像的名称（含后缀），指标]
def save_with_IQA(result_path,corrected,imgdir,psnr_str,ssim_str,uiqm_str,uciqe_str):
        # 计算完成后标有指标数值的图片存储路径        
        # ex='GDCP PSNR_SSIM_UIQM_UCIQE' # 文件夹名称        
        # ex='UColor-ReTrain PSNR_SSIM_UIQM_UCIQE' # 文件夹名称        
        # ex='WaterNet-ReTrain PSNR_SSIM_UIQM_UCIQE' # 文件夹名称        
        ex='PSNR_SSIM_UIQM_UCIQE-FV5-110-remark' # 文件夹名称
        # ex='PSNR_SSIM_UIQM_UCIQE-upgrade7_5-R90-remark' # 文件夹名称
        res_folder = os.path.join(result_path, ex)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        # 在图片上绘制文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(corrected.shape[1] * 0.8), int(corrected.shape[0] * 0.75)) # 宽*0.8处 高*0.75处
         # 根据图像宽度动态调整字体大小
        fontScale = corrected.shape[0] / 1000  # 这里的800可以根据需要调整
        fontColor = (255, 255, 255)  # 白色
        lineType = 2

        cv2.putText(corrected, 'PSNR: ' + psnr_str, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(corrected, 'SSIM: ' + ssim_str, 
                    (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + 30),  # 0水平 1垂直 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(corrected, 'UIQM: ' + uiqm_str, 
                    (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + 60),  # 0水平 1垂直 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(corrected, 'UCIQE: ' + uciqe_str, 
                    (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + 90),  # 0水平 1垂直 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
                   
        # 拼接保存图片的路径
        res_img_path = os.path.join(result_path, ex, imgdir)
        
        # 检查同名文件是否已存在，如果存在，则删除它
        if os.path.exists(res_img_path):
            os.remove(res_img_path)

        # 保存带有指标的图片到 res 文件夹内，注意与读取时一致（io读则io.imsave保存RGB，cv2读则cv2.imwrite保存BGR）
        cv2.imwrite(res_img_path, corrected)


def main():
    reference_path = sys.argv[1]

    reference_dirs = os.listdir(reference_path)

    N=0
    imagenum=1 # 目前处理图片是第几张
    for imgdir in reference_dirs:
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
            cv2.imwrite(res_img_path, reference)

    

if __name__ == '__main__':
    main()
