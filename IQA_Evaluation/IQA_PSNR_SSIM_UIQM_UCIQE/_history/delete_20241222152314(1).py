# 删除 某文件夹内 指定命名格式图片/删除不符合某命名格式的图片
import os
import glob
import fnmatch

def delete_io_png_files(directory):
    # 构建文件路径模式
    file_pattern = os.path.join(directory, "*.png")
    # file_pattern = os.path.join(directory, "*.jpg")
    # file_pattern = os.path.join(directory, "*.jpeg")
    # file_pattern = os.path.join(directory, "*_out.png")
    # 获取所有匹配的文件路径
    files_to_delete = glob.glob(file_pattern)
    # 过滤掉以 _out.png 结尾的文件
    files_to_delete = [f for f in files_to_delete if not fnmatch.fnmatch(f, '*_out.png')]
    # files_to_delete = [f for f in files_to_delete if not fnmatch.fnmatch(f, '*_out.jpg')]
    # files_to_delete = [f for f in files_to_delete if not fnmatch.fnmatch(f, '*_out.jpeg')]
    # 现在 files_to_delete 包含所有以 .png 结尾但不以 _out.png 结尾的文件
    
    # 初始化删除计数
    deleted_count = 0

    # 遍历文件路径并删除文件
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    return deleted_count

# 指定文件夹路径
directory_path = r"./ToChange/MSE+0.0015FFL-C60"
# directory_path = r"D:\Desktop\test\\New4_4_C1_4-256-110"
deleted_files_count = delete_io_png_files(directory_path)
print(f"Total deleted files: {deleted_files_count}")