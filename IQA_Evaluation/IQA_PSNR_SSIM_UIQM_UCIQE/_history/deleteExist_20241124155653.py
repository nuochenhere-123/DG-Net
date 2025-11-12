import os

def delete_common_files(raw_exist_dir, raw_needToDeleteIn_dir):
    # 获取 raw-exist 文件夹中的所有文件名（仅保留文件名，不包括路径）
    raw_exist_files = set(os.listdir(raw_exist_dir))
    
    # 获取 raw-needToDeleteIn 文件夹中的所有文件名（仅保留文件名，不包括路径）
    raw_needToDeleteIn_files = set(os.listdir(raw_needToDeleteIn_dir))
    
    # 找出在 raw-exist 中存在的文件
    common_files = raw_exist_files.intersection(raw_needToDeleteIn_files)
    
    # 打印需要删除的文件列表
    print(f"Files to delete from {raw_needToDeleteIn_dir}:")        
    sum=1
    # 从 raw-needToDeleteIn 文件夹中删除这些文件
    for file in common_files:
        print(str(sum)+": "+file)
        file_path = os.path.join(raw_needToDeleteIn_dir, file)
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
            sum+=1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# 定义文件夹路径
raw_exist_dir = "./110" # 删除其他文件夹内和raw_exist_dir重复的图片
raw_needToDeleteIn_dir = "./raw-780" # 需要删除图片的所在文件夹路径

# 调用函数删除文件
delete_common_files(raw_exist_dir, raw_needToDeleteIn_dir)