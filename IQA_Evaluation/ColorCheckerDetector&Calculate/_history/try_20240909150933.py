

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

    _,h ,w , c = image_test.shape
    # print(result_h.shape)
    for id in range(0,1):
        result_h0 = (image_test[id] * 255).astype(np.uint8)
        # print(result_h0.shape)
        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        #图片名称为imgdir，此处提取前缀
        imgname_cor = self.test_image_name.split('.')[0]
        image_path1 = os.path.join(image_path0, imgname_cor+'_out.'+self.test_image_name.split('.')[1])
        print(str(self.counter)+": "+image_path1)
        io.imsave(image_path1, result_h0) 
        self.counter += 1