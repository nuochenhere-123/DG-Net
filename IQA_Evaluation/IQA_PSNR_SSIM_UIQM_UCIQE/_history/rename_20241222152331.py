import os

def rename_images_in_directory(directory_path, original_suffix, new_suffix):
    """
    重命名目录中的图像文件，去除指定的后缀部分。
    
    :param directory_path: 图像文件所在的目录路径。
    :param original_suffix: 需要去除的文件名后缀部分。
    :param new_suffix: 新的文件名后缀部分（如没有则传入空字符串）。
    """
    files = os.listdir(directory_path)
    image_num = 1
    
    for filename in files:
        if filename.endswith(('.png', '.jpg')):  # 检查文件是否为图像
            if original_suffix in filename:
                # 提取新的文件名
                new_name = filename.replace(original_suffix, new_suffix)
                # 输出文件号和新的文件名
                print(f"{image_num}: {new_name}")
                # 构建完整的源文件和目标文件路径
                source_path = os.path.join(directory_path, filename)
                target_path = os.path.join(directory_path, new_name)
                # 检查是否目标文件名已经存在
                if not os.path.exists(target_path):
                    os.rename(source_path, target_path)
                    image_num += 1
                else:
                    print(f"目标文件名 {target_path} 已经存在，跳过重命名。")
            else:
                print(f"文件 {filename} 不包含指定的后缀部分，跳过重命名。")


if __name__ == "__main__":
    # 需要重命名的图像所在目录
    result_path = "./ToChange/WO-WeightMap-C60"
    # 原始文件名中需要去除的后缀
    original_suffix = "_out"
    # 新的文件名后缀（空字符串表示不添加新的后缀）
    new_suffix = "_WO-WeightMAP"
    
    rename_images_in_directory(result_path, original_suffix, new_suffix)
