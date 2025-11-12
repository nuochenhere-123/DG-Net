import cv2
import math
import numpy as np
import os


def uicm(img):
    b, r, g = cv2.split(img)
    RG = r - g
    YB = (r + g) / 2 - b
    m, n, o = np.shape(img)  # img为三维 rbg为二维
    K = m * n
    alpha_L = 0.1
    alpha_R = 0.1  ##参数α 可调
    T_alpha_L = math.ceil(alpha_L * K)  # 向上取整
    T_alpha_R = math.floor(alpha_R * K)  # 向下取整

    RG_list = RG.flatten()
    RG_list = sorted(RG_list)
    sum_RG = 0
    for i in range(T_alpha_L + 1, K - T_alpha_R):
        sum_RG = sum_RG + RG_list[i]
    U_RG = sum_RG / (K - T_alpha_R - T_alpha_L)
    squ_RG = 0
    for i in range(K):
        squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
    sigma2_RG = squ_RG / K

    YB_list = YB.flatten()
    YB_list = sorted(YB_list)
    sum_YB = 0
    for i in range(T_alpha_L + 1, K - T_alpha_R):
        sum_YB = sum_YB + YB_list[i]
    U_YB = sum_YB / (K - T_alpha_R - T_alpha_L)
    squ_YB = 0
    for i in range(K):
        squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
    sigma2_YB = squ_YB / K

    Uicm = -0.0268 * np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586 * np.sqrt(sigma2_RG + sigma2_YB)
    return Uicm


def EME(rbg, L):
    m, n = np.shape(rbg)  # 横向为n列 纵向为m行
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)
    # A1 = np.zeros((L, L))
    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1 + L, n1:n1 + L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))

            if rbg_min > 0:
                rbg_ratio = rbg_max / rbg_min
            else:
                rbg_ratio = rbg_max  ###
            E = E + np.log(rbg_ratio + 1e-5)

            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2 * E / (number_m * number_n)
    return E_sum


def UICONM(rbg, L):  # wrong
    m, n, o = np.shape(rbg)  # 横向为n列 纵向为m行
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)
    A1 = np.zeros((L, L))  # 全0矩阵
    m1 = 0
    logAMEE = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1 + L, n1:n1 + L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))
            plip_add = rbg_max + rbg_min - rbg_max * rbg_min / 1026
            if 1026 - rbg_min > 0:
                plip_del = 1026 * (rbg_max - rbg_min) / (1026 - rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del / plip_add
                    local_b = math.log(plip_del / plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026 - 1026 * ((1 - logAMEE / 1026) ** (1 / (number_n * number_m)))
    return logAMEE


if __name__ == '__main__':
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
        if  imgdir.endswith('_out.png'): # 文件后缀
            #corrected image 注意与保存时一致（io读则io保存RGB，cv2读则cv2保存BGR）
            corrected = cv2.imread(os.path.join(result_path,imgdir))
            #图片名称为imgdir，此处提取前缀
            imgname_cor = imgdir.split('.')[0]
            #第几张：图片名称
            print(imagenum,":",imgname_cor)
            #计算指标
            r, b, g = cv2.split(corrected)

            Uicm = uicm(corrected)

            EME_r = EME(r, 8)
            EME_b = EME(b, 8)
            EME_g = EME(g, 8)
            Uism = 0.299 * EME_r + 0.144 * EME_b + 0.557 * EME_g

            Uiconm = UICONM(corrected, 8)

            uiqm = 0.0282 * Uicm + 0.2953 * Uism + 0.6765 * Uiconm
            # 更新数值
            imagenum=imagenum+1
            sumuiqm += uiqm
            N +=1
            # # 记录单张图片结果
            # with open(os.path.join(result_path, save_txt_name), 'a') as f:
            #     f.write('{}: uiqm={}, uciqe={}\n'.format(imgdir,uiqm,uciqe))
            # # 将 指标 数值转换为字符串格式，保留四位小数
            # uciqe_str = "{:.4f}".format(uciqe)
            # uiqm_str = "{:.4f}".format(uiqm)
            # # 保存带指标图片至res_folder中,corrected为原需计算指标的图片
            # save_with_IQA(result_path,corrected,imgdir,uciqe_str,uiqm_str)

    # 记录图片结果均值
    # muciqe = sumuciqe/N
    muiqm = sumuiqm/N
    # # 将 指标 数值转换为字符串格式，保留六位小数
    # muciqe_str = "{:.6f}".format(muciqe)
    # muiqm_str = "{:.6f}".format(muiqm)
    # with open(os.path.join(result_path, save_txt_name), 'a') as f:
    #     f.write('\nAverage: uiqm={}, uciqe={}\n\n'.format(muiqm, muciqe))
    print("Sum:{} Average: uiqm={}".format(N, muiqm))
    
    
    # # 重命名文件名
    # # save_txt_name_new = '000_UIQM-{}_UCIQE-{}_Orginal.txt'.format(muiqm_str, muciqe_str)
    # # save_txt_name_new = '000_UIQM-{}_UCIQE-{}_epoch50.txt'.format(muiqm_str, muciqe_str)
    # save_txt_name_new = 'FV1-49-ep50-UIQM-{}-UCIQE-{}.txt'.format(muiqm_str, muciqe_str)
    # os.rename(os.path.join(result_path, save_txt_name),os.path.join(result_path, save_txt_name_new))       