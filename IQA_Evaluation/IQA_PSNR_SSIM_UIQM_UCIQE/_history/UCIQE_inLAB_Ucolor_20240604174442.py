'''  
水下无参考质量评价指标: UCIQE
(Ucolor版)(如: Ucolor中Test-C60的平均UCIQE为0.48, 本代码结果为0.466107)

Usage:
 $ python UCIQE_Ucolor_inLAB.py RESULT_PATH
eg. python UCIQE_inLAB_Ucolor.py .\RESULT_PATH

保存带指标图片 至 图片路径的 'Res_UCIQE_Ucolor_inLAB' 文件夹 中（第63行ex变量）
'''
# SUCCESS for usage
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os
import sys
import cv2

def getUCIQE(img):
    # UCIQE Lab色彩空间
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    img = (img*255).astype('uint8')
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0
   
    #1st term 色度的标准差
    Img_Chr = np.sqrt(np.square(img_a) + np.square(img_b))
    Aver_Chr = np.mean(Img_Chr)
    # 标准差：距离均值的差的平方，取均值后开平方
    Var_Chr = np.sqrt(np.mean((np.abs(1-(Aver_Chr/Img_Chr)**2))))
    
    
    #3rd term 饱和度的均值：
    # 法一：将 色度（Img_chr）除以 亮度（img_lum）和 色度（Img_chr）的 平方和的平方根（范数）
    # 用来衡量颜色的饱和度
    # 更侧重于色度与亮度之间的关系，因此可能更加准确地反映颜色的饱和度
    Img_Sat = Img_Chr/np.sqrt(Img_Chr**2+img_lum**2)
    Aver_Sat = np.mean(Img_Sat)
    # 法二：或使用标准差作为饱和度的近似度量
    # a_std = np.std(img_a)
    # b_std = np.std(img_b)
    # us = (a_std + b_std) / 2


    #2nd term 亮度的对比：最亮的第1%的值-最暗的第1%的值
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum) # 对img_lum进行排序，并返回从小到大排序后的索引数组
    top_index = sorted_index[int(len(img_lum) * 0.99)] # 最亮的第1%个像素（值）对应在img_lum中的索引值
    bottom_index = sorted_index[int(len(img_lum) * 0.01)] # 最暗的第1%个像素（值）对应在img_lum中的索引值
    con_lum = img_lum[top_index] - img_lum[bottom_index]
    
    uciqe = c1 * Var_Chr + c2 * con_lum + c3 * Aver_Sat
    print("UCIQE:", uciqe)
    
    return uciqe


# 保存带指标的图片到原路径下文件夹内[输入图像所在文件夹，输入图像，输入图像的名称（含后缀），指标]
def save_with_IQA(result_path,corrected,imgdir,uciqe_str):
        # 计算完成后标有指标数值的图片存储路径
        ex='Res_UCIQE_Ucolor_inLAB' # 文件夹名称
        res_folder = os.path.join(result_path, ex)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        # 在图片上绘制文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(corrected.shape[1] * 0.8), int(corrected.shape[0] * 0.9)) # 宽*0.8处 高*0.9处
         # 根据图像宽度动态调整字体大小
        fontScale = corrected.shape[0] / 1000  # 这里的800可以根据需要调整
        fontColor = (255, 255, 255)  # 白色
        lineType = 2

        cv2.putText(corrected, 'UCIQE: ' + uciqe_str,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        # cv2.putText(corrected, 'UCIQE: ' + uciqe_str, 
        #             (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + 30), # 0水平 1垂直 
        #             font, 
        #             fontScale,
        #             fontColor,
        #             lineType)

        # 拼接保存图片的路径
        res_img_path = os.path.join(result_path, ex, imgdir)
        
        # 检查同名文件是否已存在，如果存在，则删除它
        if os.path.exists(res_img_path):
            os.remove(res_img_path)

        # 保存带有指标的图片到 res 文件夹内，注意与读取时一致（io读则io.imsave保存RGB，cv2读则cv2.imwrite保存BGR）
        cv2.imwrite(res_img_path, corrected)
        

def main():
    # 需要计算指标的图片所在路径
    result_path = sys.argv[1]
    result_dirs = os.listdir(result_path)

    sumuciqe = 0.
    N=0
    imagenum=1 # 目前处理图片是第几张
    for imgdir in result_dirs:
        # if '.png' in imgdir or '.jpg' in imgdir : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('.png_out.png'):  # 文件夹内固定命名格式的图片
        if  imgdir.endswith('original.png'): # 文件后缀
            #corrected image 注意与保存时一致（io读则io保存RGB，cv2读则cv2保存BGR）
            corrected = cv2.imread(os.path.join(result_path,imgdir))
            #图片名称为imgdir，此处提取前缀
            imgname_cor = imgdir.split('.')[0]
            #第几张：图片名称
            print(imagenum,":",imgname_cor)
            #计算指标
            uciqe = getUCIQE(corrected)
            # 更新数值
            imagenum=imagenum+1
            sumuciqe += uciqe
            N +=1
            # 记录单张图片结果
            with open(os.path.join(result_path,'Orginal_UCIQE_Ucolor_inLAB.txt'), 'a') as f:
                f.write('{}: uciqe={}\n'.format(imgdir,uciqe))
            # 将 指标 数值转换为字符串格式，保留四位小数
            uciqe_str = "{:.4f}".format(uciqe)
            # 保存带指标图片至res_folder中,corrected为原需计算指标的图片
            save_with_IQA(result_path,corrected,imgdir,uciqe_str)

    # 记录图片结果均值
    muciqe = sumuciqe/N
    with open(os.path.join(result_path,'Orginal_UCIQE_Ucolor_inLAB.txt'), 'a') as f:
        f.write('\nAverage: uciqe={}\n\n'.format(muciqe))
    print("Sum:{} Average: uciqe={}".format(N, muciqe))

if __name__ == '__main__':
    main()