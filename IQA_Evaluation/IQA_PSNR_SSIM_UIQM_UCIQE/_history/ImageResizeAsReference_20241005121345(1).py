

if __name__ == '__main__':
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