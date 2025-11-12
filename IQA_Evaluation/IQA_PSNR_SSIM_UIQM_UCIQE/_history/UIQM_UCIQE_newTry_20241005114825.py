# 水下无参考质量评价指标: UIQM & UCIQE
# (Ucolor版)(如: Ucolor中Test-C60的平均UCIQE为0.48, 本代码结果为0.466107)

# Usage:
#  $ python UCIQE_Ucolor_inLAB.py RESULT_PATH

# run
# eg. python UCIQE_UIQM_inLAB.py .\RESULT_PATH
#     python UCIQE_UIQM_inLAB.py .\upgrade2_3-U45
# 每次运行修改：1. 保存文件夹名   2. 处理图片命名格式   3. 保存txt文件名

# 保存带指标图片 至 图片路径的 'Res_UCIQE_Ucolor_inLAB' 文件夹 中（第63行ex变量）

# SUCCESS for usage
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import color, filters
import math
import os
import sys
import cv2
import skimage


# UCIQE

def getUCIQE(rgb_in):
    # calculate Chroma
    rgb_in = cv2.normalize(rgb_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    (l,a,b)=cv2.split(rgb_in)
    Chroma = np.sqrt(a*a + b*b)
    StdVarianceChroma = np.std(np.reshape(Chroma[:,:],(-1,1)))

    hsv = skimage.color.rgb2hsv(rgb_in)
    Saturation = hsv[:,:,2]
    MeanSaturation = np.mean(np.reshape(Saturation[:,:],(-1,1)))

    ContrastLuminance = max(np.reshape(l[:,:],(-1,1))) - min(np.reshape(l[:,:],(-1,1)))
    UCIQE = 0.4680 * StdVarianceChroma + 0.2745 * ContrastLuminance + 0.2576 * MeanSaturation
    return float(UCIQE)

# UCIQE

# UIQM
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    # get eme for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

##########################################################################################

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

# UIQM


# 保存带指标的图片到原路径下文件夹内[输入图像所在文件夹，输入图像，输入图像的名称（含后缀），指标]
def save_with_IQA(result_path,corrected,imgdir,uciqe_str,uiqm_str):
        # 计算完成后标有指标数值的图片存储路径
        # ex='Original_UCIQE' # 文件夹名称
        # ex='Res_UCIQE_epoch50' # 文件夹名称
        ex='New4_4_C1-C60-ep137' # 文件夹名称
        # ex='UColor-ReTrain UIQM_UCIQE' # 文件夹名称
        # ex='HFM-paper UIQM_UCIQE' # 文件夹名称
        # ex='WaterNet-ReTrain UIQM_UCIQE' # 文件夹名称
        # ex='U_Shape_Transformer-PreTrain UIQM_UCIQE' # 文件夹名称
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
        # if '.png' in imgdir or '.jpg' in imgdir : # 文件夹内所有图片
        # if imgdir.endswith('.png') and not imgdir.endswith('io.png'):  # 文件夹内固定命名格式的图片
        # if  imgdir.endswith('original.png'): # 文件后缀
        if  imgdir.endswith('_out.png') or imgdir.endswith('_out.jpg') or imgdir.endswith('_out.jpeg'): # 文件后缀
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
    save_txt_name_new = 'New4_4_C1-ep137-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'U45-UColor-ReTrain120 UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'C60-HFM-paper UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'U45-WaterNet-ReTrain UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'U45-U_Shape_Transformer-PreTrain UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    os.rename(os.path.join(result_path, save_txt_name),os.path.join(result_path, save_txt_name_new))       
    

if __name__ == '__main__':
    main()