# 用于删除 图像的宽度或高度小于某个阈值（256像素）的图片，并删除该图像文件 和 reference 中图片
import os
from PIL import Image

def delete_small_images(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    sum = 0
    min_height = 500
    min_width = 500
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 尝试打开图像文件
                # with Image.open(file_path) as img: # Image.open(fullfilename) 后没有关闭，且Image 没有close方法，在os.remove时会出现文件占用的问题
                fp = open(file_path,'rb') # 解决方案：使用文件句柄
                with Image.open(fp) as img:
                    width, height = img.size
                    # 获取最小宽高
                    if width < min_width:
                            min_width = width
                    if height < min_height:
                        min_height = height
                    # 如果图像的宽度或高度小于256像素，则删除该图像文件 和 reference 中图片
                    if width < 256 or height < 256:
                        fp.close()
                        os.remove(file_path)
                        # 删除对应的reference
                        folder_path1 = r"./reference-780"
                        # folder_path1 = r"D:\Desktop\训练集900&验证集60\reference-900"
                        file_path1 = os.path.join(folder_path1, file)
                        os.remove(file_path1)
                        sum = sum + 1
                        print(f"Deleted: {file_path} and {file_path1}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return sum, min_height, min_width

# 要删除图像的文件夹路径
folder_path = r"./raw-780"

# 调用函数删除小图像
sum, min_height, min_width = delete_small_images(folder_path)
print("成功删除 ",sum," 张图片")
# print("成功删除{}张图片".format(sum))
print("所有图片中最小高度为：",min_height," 最小宽度为：", min_width)