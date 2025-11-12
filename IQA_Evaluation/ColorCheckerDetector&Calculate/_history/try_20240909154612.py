import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf
import skimage.io as io
from PIL import Image  # for loading images as YCbCr format


# 读取单张图片
def get_image_original(image_path,is_grayscale=False):
  image = io.imread(image_path, is_grayscale)
  image = image.astype(np.float32)
  if image.shape[-1] == 4:
    # 如果图像有四个通道，假设第四个通道为 alpha 通道，只保留前三个通道（RGB）
    image = image[:, :, :3]
  return image/255.0


if __name__ == '__main__':    

# local library
# 获取 测试图像
    image_test =  get_image_original("FujiFilm_Z33_UW-Portrait.jpeg",is_grayscale=False)
    image_test0 =  get_image_original("FujiFilm_Z33_UW-Portrait.jpeg",is_grayscale=False)
    shape = image_test.shape
    shape0 = image_test0.shape
    print("original: ",shape[0],shape[1], shape[2])
    # 使图像宽高能被8整除
    RGB=Image.fromarray(np.uint8(image_test*255))
    RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    shape = image_test.shape
    print("new: ",shape[0],shape[1], shape[2])

    h , w, c = image_test.shape
    final_enhanced_img = np.clip(image_test, 0, 1).reshape(h , w , 3)
    # print(result_h0.shape)
    image_path0 = os.path.join(os.getcwd(), "./")
    #图片名称为imgdir，此处提取前缀
    imgname_cor = "FujiFilm_Z33_UW-Portrait.jpeg".split('.')[0]
    image_path1 = os.path.join(image_path0, imgname_cor+'_out.'+"FujiFilm_Z33_UW-Portrait.jpeg".split('.')[1])
    
    # plt.imsave(image_path1, final_enhanced_img)
    h0, w0, c0 = image_test0.shape
    result_h0 = (final_enhanced_img * 255).astype(np.uint8)
    new_size_res = np.array( Image.fromarray(result_h0).resize(w0,h0) )
    io.imsave(image_path1, new_size_res) 
    
    shape = new_size_res.shape
    print("final: ",shape[0],shape[1], shape[2])
        