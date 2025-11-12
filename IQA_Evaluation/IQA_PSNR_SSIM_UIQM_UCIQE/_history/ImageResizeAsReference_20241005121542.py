import cv2
import os
import sys

if __name__ == '__main__':
    
    # 需要计算指标的图片所在路径
    result_path = sys.argv[1]
    result_dirs = os.listdir(result_path)

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
            # 删除对应的reference
            folder_path1 = r"./reference-780"
            file_path1 = os.path.join(folder_path1, file)
            
     # 获取 测试图像
    image_test0 =  get_image_original(self.test_image_name,is_grayscale=False)
    original_shape = image_test0.shape
    original_size = (original_shape[1], original_shape[0])
    # resize回 原大小 保存
      result_h0 = (final_enhanced_img * 255).astype(np.uint8)
      a=Image.fromarray(result_h0)
      b=a.resize(original_size)
      original_size_enhanced = np.array(b)
      io.imsave(image_path1, original_size_enhanced) 
      print(str(self.id)+": "+image_path1)