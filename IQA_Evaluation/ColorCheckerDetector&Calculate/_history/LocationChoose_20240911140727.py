import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
import os

def CIEDE2000(Lab_1, Lab_2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    C_25_7 = 6103515625 # 25**7
    
    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))
    
    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2
    
    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)
    
    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi
    
    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_    
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi        
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)
    
    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2
    
    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_
    
    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi 
    else: h_ave = h1_ + h2_
    
    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)
    
    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))
    
    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))  
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T
    
    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1
    
    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H
    
    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00


if __name__ == '__main__':    
        
    # 读取图像
    image_path = './checker_input/Cannon_D10_UW-Portrait.jpeg'  # 替换为你的图像路径
    dir = './' # 需要保存 记录位置的文件 所在路径
    # image_path = '1.png'  # 替换为你的图像路径
    image = Image.open(image_path)
    image_np = np.array(image)
    # 显示图像
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    save_txt_name_X = "Location_X.txt"
    save_txt_name_Y = "Location_Y.txt"
    
    now = 0
    with open(os.path.join(dir,save_txt_name_X), 'a') as f:
        f.write('Start:\n')
    with open(os.path.join(dir,save_txt_name_Y), 'a') as f:
        f.write('Start:\n ')

    # 事件处理函数
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            # 获取鼠标点击位置的坐标
            x, y = int(event.xdata), int(event.ydata)
            
            # 获取指定位置的RGB值
            rgb_pixel = image_np[y, x, :3]  # 注意y在前，x在后
            print(f"点击位置 ({x}, {y}) 的 RGB 值: {rgb_pixel}")

            with open(os.path.join(dir,save_txt_name_X), 'a') as f:
                f.write('{}, '.format(x))
            with open(os.path.join(dir,save_txt_name_Y), 'a') as f:
                f.write('{}, '.format(y))
            now += 1
            if now==10:
                with open(os.path.join(dir,save_txt_name_X), 'a') as f:
                    f.write('\n')
                with open(os.path.join(dir,save_txt_name_Y), 'a') as f:
                    f.write('\n')
                now = 0
            
            # 将RGB值转换为[0, 1]范围的数组，用于转换为LAB颜色空间
            rgb_array = np.array([[rgb_pixel]], dtype=np.float32) / 255.0

            # 使用skimage将RGB转换为LAB
            lab_pixel = color.rgb2lab(rgb_array)[0, 0]
            print(f"LAB值: L={lab_pixel[0]:.2f}, a={lab_pixel[1]:.2f}, b={lab_pixel[2]:.2f}")

            # 计算评分
            # # dark skin
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (37.99, 13.56, 14.06)))
            # # light skin
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (65.71, 18.13, 17.81)))
            # # blue sky
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (49.93, -4.88, -21.93)))
            # # foliage
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (43.14, -13.10, 21.91)))
            # # blue flower
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (55.11, 8.84, -25.40)))
            # # bluish green
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (70.72, -33.40, -0.199)))
            
            # # orange
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (62.66, 36.07, 57.10)))
            # # purplish blue
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (40.02, 10.41, -45.96)))
            # # moderate red
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (51.12, 48.24, 16.25)))
            # # purple
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (30.33, 22.98, -21.59)))
            # yellow green
            print("CIEDE2000:", CIEDE2000(lab_pixel, (72.53, -23.71, 57.26)))
            # # orange yellow
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (71.94, 19.36, 67.86)))
            
            # # blue
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (28.78, 14.18, -50.30)))
            # # green
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (55.26, -38.34, 31.37)))
            # # red
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (42.10, 53.38, 28.19)))
            # # yellow
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (81.73, 4.04, 79.82)))
            # # magenta
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (51.94, 49.99, -14.57)))
            # # cyan
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (51.04, -28.63, -28.64)))
            
            # # white 9.5(.05D)
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (96.54, -0.425, 1.186)))
            # # neutral 8(.23D)
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (81.26, -0.638, -0.335)))
            # # neutral 6.5(.44D)
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (66.77, -0.734, -0.504)))
            # # neutral 5(.70D)
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (50.87, -0.153, -0.270)))
            # # neutral 3.5(1.05D)
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (35.66, -0.421, -1.231)))
            # # black 2(1.5D)
            # print("CIEDE2000:", CIEDE2000(lab_pixel, (20.46, -0.079, -0.973)))

   
    # 连接鼠标点击事件
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # 显示图像窗口并等待用户操作
    plt.show()