# 水下无参考质量评价指标: UIQM & UCIQE
# (Ucolor版)(如: Ucolor中Test-C60的平均UCIQE为0.48, 本代码结果为0.466107)

# Usage:
#  $ python UCIQE_Ucolor_inLAB.py RESULT_PATH

# run
# eg. python UCIQE_UIQM_inLAB.py .\RESULT_PATH
#     python UCIQE_UIQM_inLAB.py .\upgrade2_3-U45
#     python UCIQE_UIQM_inLAB.py ./RetinexBased-C60    
# 每次运行修改：1. 保存文件夹名   2. 处理图片命名格式   3. 保存txt文件名

# 保存带指标图片 至 图片路径的 'Res_UCIQE_Ucolor_inLAB' 文件夹 中（第215行ex变量）

# SUCCESS for usage
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import color, filters
import math
import os
import sys
import cv2


# UCIQE
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
# UCIQE

# UIQM
def getUIQM(a):
    rgb = a
    gray = color.rgb2gray(a)
   
    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int32(al1 * len(rgl))
    T2 = np.int32(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    print("UIQM:", uiqm)
    
    return uiqm

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)
            if bottom == 0:
                m = 0
            else:
                m = top/bottom

            if m ==0.:
                s+=0
            else:
                s += (m) * np.log(m)

    return plipmult(w,s)
# UIQM


# 保存带指标的图片到原路径下文件夹内[输入图像所在文件夹，输入图像，输入图像的名称（含后缀），指标]
def save_with_IQA(result_path,corrected,imgdir,uciqe_str,uiqm_str):
        # 计算完成后标有指标数值的图片存储路径
        # ex='Original_UCIQE' # 文件夹名称
        # ex='Res_UCIQE_epoch50' # 文件夹名称
        # ex='New4_4_C1_4-UIQM_UCIQE-ep193' # 文件夹名称
        # ex='WO-SCA-UIQM_UCIQE-ep137' # 文件夹名称
        # ex='GDCP-UIQM_UCIQE' # 文件夹名称
        # ex='Fusion-UIQM_UCIQE' # 文件夹名称
        ex='RetinexBased-UIQM_UCIQE' # 文件夹名称
        # ex='ZSRM-UIQM_UCIQE' # 文件夹名称
        # ex='WWPF-UIQM_UCIQE' # 文件夹名称
        # ex='PCFB-UIQM_UCIQE' # 文件夹名称
        # ex='Ucolor-UIQM_UCIQE' # 文件夹名称
        # ex='WaterNet-UIQM_UCIQE' # 文件夹名称
        # ex='DICAM-UIQM_UCIQE' # 文件夹名称
        # ex='UShapeTransformer-UIQM_UCIQE' # 文件夹名称
        # ex='FUnIE-ReTrain-PyTorch-UIQM_UCIQE' # 文件夹名称
        # ex='PUGAN-Pretrain-PyTorch-UIQM_UCIQE' # 文件夹名称
        # ex='UWMamba-Pretrain-PyTorch-UIQM_UCIQE' # 文件夹名称
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
        cv2.putText(corrected, 'UIQM: ' + uiqm_str, 
                    (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + 30), # 0水平 1垂直 
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
    # 需要计算指标的图片所在路径
    result_path = sys.argv[1]
    result_dirs = os.listdir(result_path)

    sumuciqe = 0.
    sumuiqm = 0.
    save_txt_name = "save_txt_name.txt"

    N=0
    imagenum=1 # 目前处理图片是第几张
    for imgdir in result_dirs:
        if '.png' in imgdir or '.jpg' in imgdir : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('io.png'):  # 文件夹内固定命名格式的图片
        # if  imgdir.endswith('original.png'): # 文件后缀
        # if  imgdir.endswith('_out.png') or imgdir.endswith('_out.jpg') or imgdir.endswith('_out.jpeg'): # 文件后缀
            #corrected image 注意与保存时一致（io读则io保存RGB，cv2读则cv2保存BGR）
            corrected = cv2.imread(os.path.join(result_path,imgdir))
            #图片名称为imgdir，此处提取前缀
            imgname_cor = imgdir.split('.')[0]
            #第几张：图片名称
            print(imagenum,":",imgname_cor)
            #计算指标
            uiqm = getUIQM(corrected)
            uciqe = getUCIQE(corrected)
            # 更新数值
            imagenum=imagenum+1
            sumuciqe += uciqe
            sumuiqm += uiqm
            N +=1
            # 记录单张图片结果
            with open(os.path.join(result_path, save_txt_name), 'a') as f:
                f.write('{}: uiqm={}, uciqe={}\n'.format(imgdir,uiqm,uciqe))
            # 将 指标 数值转换为字符串格式，保留四位小数
            uciqe_str = "{:.4f}".format(uciqe)
            uiqm_str = "{:.4f}".format(uiqm)
            # 保存带指标图片至res_folder中,corrected为原需计算指标的图片
            save_with_IQA(result_path,corrected,imgdir,uciqe_str,uiqm_str)

    # 记录图片结果均值
    muciqe = sumuciqe/N
    muiqm = sumuiqm/N
    # 将 指标 数值转换为字符串格式，保留六位小数
    muciqe_str = "{:.6f}".format(muciqe)
    muiqm_str = "{:.6f}".format(muiqm)
    with open(os.path.join(result_path, save_txt_name), 'a') as f:
        f.write('\nAverage: uiqm={}, uciqe={}\n\n'.format(muiqm, muciqe))
    print("Sum:{} Average: uiqm={}, uciqe={}".format(N, muiqm, muciqe))
    
    
    # 重命名文件名
    # save_txt_name_new = '000_UIQM-{}_UCIQE-{}_Orginal.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = '000_UIQM-{}_UCIQE-{}_epoch50.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'New4_4_C7-ep137-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-WO-SCA-ep137-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'UIQS-GDCP-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-Fusion-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    save_txt_name_new = 'OceanDark-RetinexBased-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-ZSRM-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'U45-WWPF-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'U45-PCFB-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-256-Ucolor-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'UCCS-DICAM-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'UIQS-WaterNet-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'UCCS-resize-UShape_Transformer-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-FUnIE-ReTrain-Pytorch-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-resize-PUGAN-Pretrain-Pytorch-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'UIQS-256-UWMamba-Pretrain-Pytorch-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    os.rename(os.path.join(result_path, save_txt_name),os.path.join(result_path, save_txt_name_new))       
    

if __name__ == '__main__':
    main()