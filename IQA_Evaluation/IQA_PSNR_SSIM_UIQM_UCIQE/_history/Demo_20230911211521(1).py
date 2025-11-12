import math
import os

import cv2
import natsort
import numpy as np
import xlwt
# 注意，这里的xlwt是python的第三方模块，需要下载安装才能使用，不然导入不了
# （python第三方库的安装也非常简单，
# 打开命令行，输入pip install xlwt就可以了）


def UCIQE(x):
    img = x  # 图片路径
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # RGB转为HSV
    H, S, V = cv2.split(hsv)
    delta = np.std(H) / 180  # 色度的标准差
    mu = np.mean(S) / 255  # 饱和度的平均值
    n, m = np.shape(V)
    number = math.floor(n * m / 100)  # 所需像素的个数
    Maxsum, Minsum = 0, 0
    V1, V2 = V / 255, V / 255

    for i in range(1, number + 1):
        Maxvalue = np.amax(np.amax(V1))
        x, y = np.where(V1 == Maxvalue)
        Maxsum = Maxsum + V1[x[0], y[0]]
        V1[x[0], y[0]] = 0

    top = Maxsum / number

    for i in range(1, number + 1):
        Minvalue = np.amin(np.amin(V2))
        X, Y = np.where(V2 == Minvalue)
        Minsum = Minsum + V2[X[0], Y[0]]
        V2[X[0], Y[0]] = 1

    bottom = Minsum / number

    conl = top - bottom

    ###对比度
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2575 * mu
    # print(delta, conl, mu)
    # print(uciqe)
    return uciqe



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




folder = r"C:/Cworkspace/202309"
# folder_F = r"D:/DSIRF/Mcode/Result"
# folder = "C:/Users/Administrator/Desktop/Databases/Dataset"
path = folder + "/OceanDark-480"
# path1 = folder_F+"/Fusion_GDCP_IBLA"
files = os.listdir(path)
files = natsort.natsorted(files)

# book = xlwt.Workbook(encoding='utf-8', style_compression=0)
# # 调用xlwt模块中的Workbook方法来创建一个excel表格类型文件，其中的第一个参数是设置数据的编码格式，
# # 这里是’utf - 8’的形式，style_compression设置是否压缩，不是很常用，赋值为0表示不压缩。
# sheet = book.add_sheet('test1',cell_overwrite_ok=True)
# # 用book对象调用add_sheet方法来建立一张sheet表，这里面的第一个参数很明显
# # 就是设置sheet表格的名称，第二个参数cell_overwrite_ok用于确认同一个cell
# # 单元是否可以重设值，这里赋值为True就表示可重设值。
# col = ('ImageName','UCIQE','UIQM')
# # 用一个元组col自定义列的数量以及各列的属性名，比如我这里是8列，列属性名有“电影详情链接”，“图片链接”等。
# for j in range(0,3):
#         sheet.write(0,3,col[j])
# 很简单，用一个for循环将col元组的元组值（也就是列属性名）写入到sheet表单中。这里调用的是write方法，该方法的第一个参数是行、第二个参数是列、第三个当然就是col元组值。因为这里写进去的是列名，所以都是在第一行。



# files1 = os.listdir(path1)
# files1 =  natsort.natsorted(files1)

for i in range(len(files)):
    file = files[i]
    # file1 = files1[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    test_uciqe = []
    test_uiqm = []
    test_uicm = []
    test_uiconm = []
    test_uism = []
    # test_uiqm = []
    ImageName = []
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread(folder +'/OceanDark-480/' + file)
        r, b, g = cv2.split(img)
        ImageName = [ImageName,prefix]

        uciqe = UCIQE(img)
        # num_uiqm = UIQM()

        Uicm = uicm(img)

        EME_r = EME(r, 8)
        EME_b = EME(b, 8)
        EME_g = EME(g, 8)
        Uism = 0.299 * EME_r + 0.144 * EME_b + 0.557 * EME_g

        Uiconm = UICONM(img, 8)

        uiqm = 0.0282 * Uicm + 0.2953 * Uism + 0.6765 * Uiconm
        print('UICM:',Uicm)

        test_uciqe = [test_uciqe,uciqe]
        test_uiqm = [test_uiqm,uiqm]
        test_uicm = [test_uicm,Uicm]
        test_uiconm = [test_uiconm,Uiconm]
        test_uism = [test_uism,Uism]
# ImageName.to_excel()
test_uiqm.to_excel("C:/Cworkspace/202309/Metric2309/OceanDark/uiqm.xlsx")
test_uciqe.to_excel("C:/Cworkspace/202309/Metric2309/OceanDark/uciqe.xlsx")
test_uicm.to_excel("C:/Cworkspace/202309/Metric2309/OceanDark/uicm.xlsx")
test_uiconm.to_excel("C:/Cworkspace/202309/Metric2309/OceanDark/uiconm.xlsx")
test_uism.to_excel("C:/Cworkspace/202309/Metric2309/OceanDark/uism.xlsx")
# test_uciqe.to_excel("C:/Cworkspace/202309/Metric2309/OceanDark/uciqe.xlsx")



        # print('----------------------')



        # cv2.imwrite('D:/DSIRF/Mcode/TestImages/WhiteBalance/Img1/' + prefix + '_img1.jpg', img1)
        # cv2.imwrite('D:/DSIRF/Mcode/TestImages/WhiteBalance/Img2/' + prefix + '_img2.jpg', img2)
        # cv2.imwrite('D:/DSIRF/Mcode/TestImages/WhiteBalance/Img3/' + prefix + '_img3.jpg', img3)
        # cv2.imwrite('D:/DSIRF/Mcode/TestImages/WhiteBalance/Img4/' + prefix + '_img4.jpg', img4)
        # cv2.imwrite('D:/DSIRF/Mcode/TestImages/WhiteBalance/Img5/' + prefix + '_img5.jpg', img5)
