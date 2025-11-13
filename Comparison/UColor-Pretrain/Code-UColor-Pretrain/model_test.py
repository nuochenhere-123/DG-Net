from utils import ( 
  imsave,
  prepare_data
)
# wait for add to evaluate by metrics
# from ssim import *

import time
import os
import skimage.io as io
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *
from ssim import *
# local library
import rgb_lab_formulation as Conv_img
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=128,
               image_width=128,
               label_height=128, 
               label_width=128,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None, 
               sample_dir=None,
               test_image_name = None,
               test_depth_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.test_image_name = test_image_name
    self.test_depth_name = test_depth_name
    self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.c_depth_dim=c_depth_dim
    self.new_height=0
    self.new_width=0
    self.new_height_half=0 
    self.new_width_half=0
    self.new_height_half_half=0
    self.new_width_half_half=0  
    image_test =  get_image(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    RGB=Image.fromarray(np.uint8(image_test*255))
    RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    shape = image_test.shape
    self.new_height=shape[0]
    self.new_width=shape[1]


    self.build_model()

  def build_model(self):
    self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self._depth = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width,self.c_depth_dim], name='depth')
    self.pred_h = self.model()
    self.saver = tf.compat.v1.train.Saver()
     
  def train(self, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    image_test =  get_image(self.test_image_name,is_grayscale=False)
    # 去除多余通道
    if image_test.shape[-1] == 4:
      image_test = image_test[:,:,:3]
    shape = image_test.shape
    RGB=Image.fromarray(np.uint8(image_test*255))
    RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    shape = image_test.shape

    expand_test = image_test[np.newaxis,:,:,:]
#     print("expand_test:::::: ", expand_test.shape)
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
#     print("expand_zero:::::: ", expand_zero.shape)
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)
#     print("batch_test_image:::::: ", batch_test_image.shape)


    depth_test =  get_image(self.test_depth_name,is_grayscale=True)
#     # 去除多余通道
#     if depth_test.shape[-1] == 3:
#       depth_test = depth_test[:,:,:2]
    shape = depth_test.shape
    Depth=Image.fromarray(np.uint8(depth_test*255))
    Depth1=Depth.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    depth_test = np.asarray(np.float32(Depth1)/255)
#     print("depth_test: ", depth_test.shape)
    shape1 = depth_test.shape
    
    expand_test1 = depth_test[np.newaxis,:,:]
#     print("expand_test1:::::: ", expand_test1.shape)
    expand_zero1 = np.zeros([self.batch_size-1,shape1[0],shape1[1]])
#     print("expand_zero1:::::: ", expand_zero1.shape)
    batch_test_depth1 = np.append(expand_test1,expand_zero1,axis = 0)
    batch_test_depth= batch_test_depth1.reshape(self.batch_size,shape1[0],shape1[1],1)

    tf.compat.v1.global_variables_initializer().run()
    
    
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    start_time = time.time()
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image,self._depth: batch_test_depth})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)    


    _,h ,w , c = result_h.shape
    for id in range(0,1):
        result_h[id] = np.clip(result_h[id], 0, 1)
        result_h0 = (result_h[id] * 255).astype(np.uint8)
        # print(result_h0.shape)
        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        #图片名称为imgdir，此处提取前缀
        imgname_cor = self.test_image_name.split('.')[0]
        image_path1 = os.path.join(image_path0, imgname_cor+'_out.'+self.test_image_name.split('.')[1])
        print(image_path1)
        io.imsave(image_path1, result_h0) 
        

  def model(self):

    with tf.compat.v1.variable_scope("fusion_branch") as scope:

      depth_2down=max_pool_2x2(1-self._depth) 
      depth_4down=max_pool_2x2(depth_2down) 
      depth_8down=max_pool_2x2(depth_4down) 
# first HSV encoder
      HSV=tf.image.rgb_to_hsv(self.images)
      conv2_1_HSV = tf.nn.relu(conv2d(HSV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_HSV"))
      conv2_cb1_1_HSV = tf.nn.relu(conv2d(conv2_1_HSV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1_HSV"))
      conv2_cb1_2_HSV = tf.nn.relu(conv2d(conv2_cb1_1_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2_HSV"))
      conv2_cb1_3_HSV = conv2d(conv2_cb1_2_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3_HSV")
      first_add_HSV=tf.add(conv2_1_HSV,conv2_cb1_3_HSV)
      conv2_2_HSV = tf.nn.relu(conv2d(first_add_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_HSV"))     
      conv2_cb1_4_HSV = tf.nn.relu(conv2d(conv2_2_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4_HSV"))
      conv2_cb1_5_HSV = tf.nn.relu(conv2d(conv2_cb1_4_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5_HSV"))
      conv2_cb1_6_HSV =conv2d(conv2_cb1_5_HSV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_6_HSV")
      first_add1_HSV=tf.add(conv2_2_HSV,conv2_cb1_6_HSV)
      encoder1_down2_HSV=max_pool_2x2(first_add1_HSV)      

# first YUV encoder
      YUV=Conv_img.rgb_to_lab(self.images)
      conv2_1_YUV = tf.nn.relu(conv2d(YUV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_YUV"))     
      conv2_cb1_1_YUV = tf.nn.relu(conv2d(conv2_1_YUV, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1_YUV"))
      conv2_cb1_2_YUV = tf.nn.relu(conv2d(conv2_cb1_1_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2_YUV"))
      conv2_cb1_3_YUV = conv2d(conv2_cb1_2_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3_YUV")
      first_add_YUV=tf.add(conv2_1_YUV,conv2_cb1_3_YUV)
      conv2_2_YUV = tf.nn.relu(conv2d(first_add_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_YUV"))
      conv2_cb1_4_YUV = tf.nn.relu(conv2d(conv2_2_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4_YUV"))
      conv2_cb1_5_YUV = tf.nn.relu(conv2d(conv2_cb1_4_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5_YUV"))
      conv2_cb1_6_YUV =conv2d(conv2_cb1_5_YUV, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_6_YUV")
      first_add1_YUV=tf.add(conv2_2_YUV,conv2_cb1_6_YUV)
      encoder1_down2_YUV=max_pool_2x2(first_add1_YUV)   

# first RGB encoder
      conv2_1 = tf.nn.relu(conv2d(self.images, 3,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1"))
      conv2_cb1_1 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_1,conv2_1_HSV,conv2_1_YUV]), 128, 128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_1"))
      conv2_cb1_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb1_1,conv2_cb1_1_HSV,conv2_cb1_1_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_2"))
      conv2_cb1_3 = conv2d(tf.concat(axis = 3, values = [conv2_cb1_2,conv2_cb1_3_HSV,conv2_cb1_3_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_3")
      first_add=tf.add(conv2_1,conv2_cb1_3)
      conv2_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [first_add,first_add_HSV,first_add_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2"))
      conv2_cb1_4 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_2,conv2_2_HSV,conv2_2_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_4"))
      conv2_cb1_5 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb1_4,conv2_cb1_4_HSV,conv2_cb1_4_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_5"))
      conv2_cb1_6 = conv2d(tf.concat(axis = 3, values = [conv2_cb1_5,conv2_cb1_5_HSV,conv2_cb1_5_YUV]), 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb1_6")
      first_add1=tf.add(conv2_2,conv2_cb1_6)
      encoder1_down2=max_pool_2x2(first_add1)     


# second hsv encoder
      conv2_2_1_HSV = tf.nn.relu(conv2d(encoder1_down2_HSV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_1_HSV"))
      conv2_cb2_1_HSV = tf.nn.relu(conv2d(conv2_2_1_HSV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_1_HSV"))
      conv2_cb2_2_HSV = tf.nn.relu(conv2d(conv2_cb2_1_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_2_HSV"))
      conv2_cb2_3_HSV = conv2d(conv2_cb2_2_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_3_HSV")
      second_add_HSV=tf.add(conv2_2_1_HSV,conv2_cb2_3_HSV)
      conv2_2_2_HSV = tf.nn.relu(conv2d(second_add_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_2_HSV"))
      conv2_cb2_4_HSV = tf.nn.relu(conv2d(conv2_2_2_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_4_HSV"))
      conv2_cb2_5_HSV = tf.nn.relu(conv2d(conv2_cb2_4_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_5_HSV"))
      conv2_cb2_6_HSV = conv2d(conv2_cb2_5_HSV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_6_HSV")
      second_add1_HSV=tf.add(conv2_2_2_HSV,conv2_cb2_6_HSV)
      encoder2_down2_HSV=max_pool_2x2(second_add1_HSV)    

# second YUV encoder
      conv2_2_1_YUV = tf.nn.relu(conv2d(encoder1_down2_YUV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_1_YUV"))
      conv2_cb2_1_YUV = tf.nn.relu(conv2d(conv2_2_1_YUV, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_1_YUV"))
      conv2_cb2_2_YUV = tf.nn.relu(conv2d(conv2_cb2_1_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_2_YUV"))
      conv2_cb2_3_YUV = conv2d(conv2_cb2_2_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_3_YUV")
      second_add_YUV=tf.add(conv2_2_1_YUV,conv2_cb2_3_YUV)
      conv2_2_2_YUV = tf.nn.relu(conv2d(second_add_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_2_YUV"))
      conv2_cb2_4_YUV = tf.nn.relu(conv2d(conv2_2_2_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_4_YUV"))
      conv2_cb2_5_YUV = tf.nn.relu(conv2d(conv2_cb2_4_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_5_YUV"))
      conv2_cb2_6_YUV = conv2d(conv2_cb2_5_YUV, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_6_YUV")
      second_add1_YUV=tf.add(conv2_2_2_YUV,conv2_cb2_6_YUV)
      encoder2_down2_YUV=max_pool_2x2(second_add1_YUV)    



# second RGB encoder
      conv2_2_1 = tf.nn.relu(conv2d(encoder1_down2, 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_1"))
      conv2_cb2_1 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_2_1,conv2_2_1_HSV,conv2_2_1_YUV]), 3,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_1"))
      conv2_cb2_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb2_1,conv2_cb2_1_HSV,conv2_cb2_1_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_2"))
      conv2_cb2_3 = conv2d(tf.concat(axis = 3, values = [conv2_cb2_2,conv2_cb2_3_HSV,conv2_cb2_3_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_3")
      second_add=tf.add(conv2_2_1,conv2_cb2_3)
      conv2_2_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [second_add,second_add_HSV,second_add_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_2_2"))
      conv2_cb2_4 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_2_2,conv2_2_2_HSV,conv2_2_2_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_4"))
      conv2_cb2_5 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb2_4,conv2_cb2_4_HSV,conv2_cb2_4_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_5"))
      conv2_cb2_6 = conv2d(tf.concat(axis = 3, values = [conv2_cb2_5,conv2_cb2_5_HSV,conv2_cb2_5_YUV]), 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb2_6")
      second_add1=tf.add(conv2_2_2,conv2_cb2_6)
      encoder2_down2=max_pool_2x2(second_add1)    



# third hsv encoder
      conv2_3_1_HSV = tf.nn.relu(conv2d(encoder2_down2_HSV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_1_HSV"))
      conv2_cb3_1_HSV = tf.nn.relu(conv2d(conv2_3_1_HSV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_1_HSV"))
      conv2_cb3_2_HSV = tf.nn.relu(conv2d(conv2_cb3_1_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_2_HSV"))
      conv2_cb3_3_HSV = conv2d(conv2_cb3_2_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_3_HSV")
      third_add_HSV=tf.add(conv2_3_1_HSV,conv2_cb3_3_HSV)
      conv2_3_2_HSV = tf.nn.relu(conv2d(third_add_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_2_HSV"))
      conv2_cb3_4_HSV = tf.nn.relu(conv2d(conv2_3_2_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_4_HSV"))
      conv2_cb3_5_HSV = tf.nn.relu(conv2d(conv2_cb3_4_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_5_HSV"))
      conv2_cb3_6_HSV = conv2d(conv2_cb3_5_HSV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_6_HSV")
      third_add1_HSV=tf.add(conv2_3_2_HSV,conv2_cb3_6_HSV)

# third YUV encoder
      conv2_3_1_YUV = tf.nn.relu(conv2d(encoder2_down2_YUV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_1_YUV"))
      conv2_cb3_1_YUV = tf.nn.relu(conv2d(conv2_3_1_YUV, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_1_YUV"))
      conv2_cb3_2_YUV = tf.nn.relu(conv2d(conv2_cb3_1_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_2_YUV"))
      conv2_cb3_3_YUV = conv2d(conv2_cb3_2_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_3_YUV")
      third_add_YUV=tf.add(conv2_3_1_YUV,conv2_cb3_3_YUV)
      conv2_3_2_YUV = tf.nn.relu(conv2d(third_add_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_2_YUV"))
      conv2_cb3_4_YUV = tf.nn.relu(conv2d(conv2_3_2_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_4_YUV"))
      conv2_cb3_5_YUV = tf.nn.relu(conv2d(conv2_cb3_4_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_5_YUV"))
      conv2_cb3_6_YUV =conv2d(conv2_cb3_5_YUV, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_6_YUV")
      third_add1_YUV=tf.add(conv2_3_2_YUV,conv2_cb3_6_YUV)


# third RGB encoder
      conv2_3_1 = tf.nn.relu(conv2d(encoder2_down2, 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_1"))
      conv2_cb3_1 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_3_1,conv2_3_1_HSV,conv2_3_1_YUV]), 3,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_1"))
      conv2_cb3_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb3_1,conv2_cb3_1_HSV,conv2_cb3_1_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_2"))
      conv2_cb3_3 = conv2d(tf.concat(axis = 3, values = [conv2_cb3_2,conv2_cb3_2_HSV,conv2_cb3_2_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_3")
      third_add=tf.add(conv2_3_1,conv2_cb3_3)
      conv2_3_2 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [third_add,third_add_HSV,third_add_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_3_2"))
      conv2_cb3_4 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_3_2,conv2_3_2_HSV,conv2_3_2_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_4"))
      conv2_cb3_5 = tf.nn.relu(conv2d(tf.concat(axis = 3, values = [conv2_cb3_4,conv2_cb3_4_HSV,conv2_cb3_4_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_5"))
      conv2_cb3_6 =conv2d(tf.concat(axis = 3, values = [conv2_cb3_5,conv2_cb3_5_HSV,conv2_cb3_5_YUV]), 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_cb3_6")
      third_add1=tf.add(conv2_3_2,conv2_cb3_6)

      
########### concate
      third_con= tf.concat(axis = 3, values = [third_add1,third_add1_HSV,third_add1_YUV])
      channle_weight_third_con_temp=self.Squeeze_excitation_layer(third_con, out_dim=1536, ratio=16, layer_name='channle_weight_third_con_temp')
      third_con_ff = tf.nn.relu(conv2d(channle_weight_third_con_temp, 1536,512,k_h=3, k_w=3, d_h=1, d_w=1,name="third_con_ff"))

      second_con= tf.concat(axis = 3, values = [second_add1,second_add1_HSV,second_add1_YUV]) 
      channle_weight_second_con_temp=self.Squeeze_excitation_layer(second_con, out_dim=768, ratio=16, layer_name='channle_weight_second_con_temp')
      second_con_ff = tf.nn.relu(conv2d(channle_weight_second_con_temp, 768,256,k_h=3, k_w=3, d_h=1, d_w=1,name="second_con_ff"))

      first_con= tf.concat(axis = 3, values = [first_add1,first_add1_HSV,first_add1_YUV]) 
      channle_weight_first_con_temp=self.Squeeze_excitation_layer(first_con, out_dim=384, ratio=16, layer_name='channle_weight_first_con_temp')
      first_con_ff = tf.nn.relu(conv2d(channle_weight_first_con_temp, 384,128,k_h=3, k_w=3, d_h=1, d_w=1,name="first_con_ff"))
#############################################################################################################
#first decoder
      decoder_input=tf.add(third_con_ff,tf.multiply(third_con_ff,depth_4down))
      conv2_1_1_dc = tf.nn.relu(conv2d(decoder_input, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_1_dc"))
      conv2_dc1_1 = tf.nn.relu(conv2d(conv2_1_1_dc, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_1"))
      conv2_dc1_2 = tf.nn.relu(conv2d(conv2_dc1_1, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_2"))
      conv2_dc1_3 = conv2d(conv2_dc1_2, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_3")
      first_dadd=tf.add(conv2_1_1_dc,conv2_dc1_3)
      conv2_1_2_dc = tf.nn.relu(conv2d(first_dadd, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_2_dc"))
      conv2_dc1_4 = tf.nn.relu(conv2d(conv2_1_2_dc, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_4"))
      conv2_dc1_5 = tf.nn.relu(conv2d(conv2_dc1_4, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_5"))
      conv2_dc1_6 = conv2d(conv2_dc1_5, 512,512,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc1_6")
      first_dadd1=tf.add(conv2_1_2_dc,conv2_dc1_6)
      decoder1_down2=tf.compat.v1.image.resize_bilinear(first_dadd1,[second_con_ff.get_shape().as_list()[1], second_con_ff.get_shape().as_list()[2]])
     
# second decoder
      decoder_input1=tf.add(second_con_ff,tf.multiply(second_con_ff,depth_2down))
      concate_1 = tf.concat(axis = 3, values = [decoder1_down2,decoder_input1])  
      conv2_1_3_dc = tf.nn.relu(conv2d(concate_1, 512,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_3_dc"))
      conv2_dc2_1 = tf.nn.relu(conv2d(conv2_1_3_dc, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_1"))      
      conv2_dc2_2 = tf.nn.relu(conv2d(conv2_dc2_1, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_2"))
      conv2_dc2_3 = conv2d(conv2_dc2_2, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_3")
      second_dadd=tf.add(conv2_1_3_dc,conv2_dc2_3)
      conv2_1_4_dc = tf.nn.relu(conv2d(second_dadd, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_4_dc"))
      conv2_dc2_4 = tf.nn.relu(conv2d(conv2_1_4_dc, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_4"))
      conv2_dc2_5 = tf.nn.relu(conv2d(conv2_dc2_4, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_5"))
      conv2_dc2_6 = conv2d(conv2_dc2_5, 256,256,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc2_6")
      second_dadd1=tf.add(conv2_1_4_dc,conv2_dc2_6)
      decoder2_down2=tf.compat.v1.image.resize_bilinear(second_dadd1,[first_con_ff.get_shape().as_list()[1], first_con_ff.get_shape().as_list()[2]])        
# third decoder
      decoder_input2=tf.add(first_con_ff,tf.multiply(first_con_ff,(1-self._depth)))
      concate_2 = tf.concat(axis = 3, values = [decoder2_down2,decoder_input2])
      conv2_1_5_dc = tf.nn.relu(conv2d(concate_2, 256,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_5_dc"))      
      conv2_dc3_1 = tf.nn.relu(conv2d(conv2_1_5_dc, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_1"))
      conv2_dc3_2 = tf.nn.relu(conv2d(conv2_dc3_1, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_2"))
      conv2_dc3_3 = conv2d(conv2_dc3_2, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_3")
      third_dadd=tf.add(conv2_1_5_dc,conv2_dc3_3)
      conv2_1_6_dc = tf.nn.relu(conv2d(third_dadd, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_1_6_dc"))
      conv2_dc3_4 = tf.nn.relu(conv2d(conv2_1_6_dc, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_4"))
      conv2_dc3_5 = tf.nn.relu(conv2d(conv2_dc3_4, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_5"))
      conv2_dc3_6 = conv2d(conv2_dc3_5, 128,128,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_dc3_6")
      third_dadd1=tf.add(conv2_1_6_dc,conv2_dc3_6)
      conv2_refine = conv2d(third_dadd1, 128,3,k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_refine")    
 
      final_results=conv2_refine

    return final_results




  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False


  def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale
def unsqueeze2d(x, factor=2):

    x = tf.depth_to_space(x, factor)
    return x