

# 获取 测试图像
    image_test =  get_image_original(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    # 使图像宽高能被8整除
    RGB=Image.fromarray(np.uint8(image_test*255))
    RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    shape = image_test.shape
    # 为图像增加一个新的维度，使其能够与批处理张量兼容
    expand_test = image_test[np.newaxis,:,:,:]
    # 需要增加的批次量，并加上
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)
    