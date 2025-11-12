import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color

# 读取图像
image_path = 'your_image_path.jpg'  # 替换为你的图像路径
image = Image.open(image_path)
image_np = np.array(image)

# 显示图像
fig, ax = plt.subplots()
ax.imshow(image_np)

# 事件处理函数
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # 获取鼠标点击位置的坐标
        x, y = int(event.xdata), int(event.ydata)
        
        # 获取指定位置的RGB值
        rgb_pixel = image_np[y, x, :3]  # 注意y在前，x在后
        print(f"点击位置 ({x}, {y}) 的 RGB 值: {rgb_pixel}")

        # 将RGB值转换为[0, 1]范围的数组，用于转换为LAB颜色空间
        rgb_array = np.array([[rgb_pixel]], dtype=np.float32) / 255.0

        # 使用skimage将RGB转换为LAB
        lab_pixel = color.rgb2lab(rgb_array)[0, 0]
        print(f"LAB值: L={lab_pixel[0]:.2f}, a={lab_pixel[1]:.2f}, b={lab_pixel[2]:.2f}")

# 连接鼠标点击事件
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# 显示图像窗口并等待用户操作
plt.show()