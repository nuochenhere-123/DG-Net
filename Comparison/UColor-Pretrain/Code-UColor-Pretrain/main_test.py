from model_test import T_CNN
from utils import *
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pprint
import os
import PIL

flags = tf.app.flags
flags.DEFINE_integer("epoch", 120, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 128, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 128, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 128, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 128, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.0001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_pretrain", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()
def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape().as_list()  # 使用 as_list() 确保返回的是普通 Python 列表
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim  # 不再使用 dim.value
        total_parameters += variable_parameters
    print("Total parameters: {:.2f}M".format(total_parameters / 1e6))

def estimate_gflops(sess, output_tensor, input_tensor, input_shape):
    from tensorflow.python.profiler import model_analyzer
    from tensorflow.python.profiler import option_builder

    run_meta = tf.RunMetadata()
    opts = option_builder.ProfileOptionBuilder.float_operation()
    flops = model_analyzer.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    print("Total GFLOPs: {:.2f}".format(flops.total_float_ops / 1e9))  # 转为 GFLOPs

def main(_):
  #  # 设置环境变量
  # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
  # 检查是否存在可用的 GPU
  gpu_available = tf.test.is_gpu_available()

  # 设置 TensorFlow 的配置
  config = tf.ConfigProto()
  if gpu_available:
    config.gpu_options.allow_growth = True  # 允许 GPU 内存按需增长
    config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 设置 GPU 内存使用比例
    device = "/gpu:0"  # 指定使用 GPU
    print("run with GPU")
  else:
    device = "/cpu:0"  # 指定使用 CPU
    print("No GPU, run with CPU")
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  
  # 基础文件夹路径
  test_base_dir = os.path.join(os.getcwd(), 'noise-110-256')
  gdcp_base_dir = os.path.join(os.getcwd(), 'GDCP-110-256')

  # 获取子文件夹名称
  test_subfolders = sorted([f for f in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, f))])
  gdcp_subfolders = sorted([f for f in os.listdir(gdcp_base_dir) if os.path.isdir(os.path.join(gdcp_base_dir, f))])
  # print("test_subfolders: ", test_subfolders)
  # print("gdcp_subfolders: ", gdcp_subfolders)
  
  i=0
  # 遍历子文件夹
  for subfolder in test_subfolders:
    print("\nStart to test folder :", subfolder)
    data_dir = os.path.join(test_base_dir, test_subfolders[i]) ###### input image dataset
    data_dir1 = os.path.join(gdcp_base_dir, gdcp_subfolders[i]) ###### input transmission dataset
    i += 1
    print(data_dir, data_dir1)

    data = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    test_data_list = data + sorted(glob.glob(os.path.join(data_dir, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir, "*.bmp")))+sorted(glob.glob(os.path.join(data_dir, "*.jpeg")))
    test_data_list = sorted(test_data_list, key=lambda x: os.path.abspath(x)) # 绝对路径排序
    # test_names = [os.path.basename(image_path) for image_path in test_data_list]
    # print("test_data_list: ", ", ".join(test_names), "\n")
    
    data1 = sorted(glob.glob(os.path.join(data_dir1, "*.png")))
    test_data_list1 = data1 + sorted(glob.glob(os.path.join(data_dir1, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir1, "*.bmp")))+sorted(glob.glob(os.path.join(data_dir1, "*.jpeg")))
    test_data_list1 = sorted(test_data_list1, key=lambda x: os.path.abspath(x)) # 绝对路径排序
    # GDCP_names = [os.path.basename(image1_path) for image1_path in test_data_list1]
    # print("GDCP_data_list1", ", ".join(GDCP_names), "\n")
    
    total_time = 0.0
    num = 0
    for ide in range(0,len(test_data_list)):
      num += 1
      image_test1 =  get_image(test_data_list[ide],is_grayscale=False)
      shape = image_test1.shape
      RGB=Image.fromarray(np.uint8(image_test1*255))
      RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
      image_test = np.asarray(np.float32(RGB1)/255)
      depth_test1 =  get_image(test_data_list1[ide],is_grayscale=False)
      Depth=Image.fromarray(np.uint8(depth_test1*255))
      Depth1=Depth.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
      depth_test = np.asarray(np.float32(Depth1)/255)

      print("正在处理第", ide+1, "/", len(test_data_list), "张图片......")
      shape = image_test.shape
      tf.reset_default_graph()
      with tf.Session() as sess:
        # with tf.device('/cpu:0'):
        with tf.device(device):
          srcnn = T_CNN(sess, 
                    image_height=shape[0],
                    image_width=shape[1],  
                    label_height=FLAGS.label_height, 
                    label_width=FLAGS.label_width, 
                    batch_size=FLAGS.batch_size,
                    c_dim=FLAGS.c_dim, 
                    c_depth_dim=FLAGS.c_depth_dim,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir,
                    test_image_name = test_data_list[ide],
                    test_depth_name = test_data_list1[ide],
                    id = ide
                    )
          count_parameters()
          start_time = time.time()
          srcnn.train(FLAGS)
          end_time = time.time()
          total_time += end_time - start_time
          print("Inference time: {:.4f} seconds".format(end_time - start_time))
          # 🔽 添加：统计 GFLOPs（注意：模型必须已构建）
          input_tensor = tf.placeholder(tf.float32, shape=[1, shape[0], shape[1], 3])
          output_tensor = srcnn.pred_h  # 请确认这个是模型输出
          estimate_gflops(sess, output_tensor, input_tensor, [1, shape[0], shape[1], 3])

          sess.close()
      tf.get_default_graph().finalize()
    print("Total time: ", total_time, ", inference time: ", total_time/num)

      
      
if __name__ == '__main__':

  tf.app.run()
