### Mamba-ssm 源文件：[见 Github](https://github.com/state-spaces/mamba "见 Github")  
### UWMamba 源码： [见 Github](https://github.com/LeonSakura/UWMamba)

## 测试
1. 修改 **预训练模型位置** ：
     > **./experiments**/UIEB/**UWMamba.yaml** 内MODEL：PRETRAINED：（Line 21）  

2. 修改 **输入网络的图像尺寸** ：  
    > **./lib**/datasets/**UIEB.py** 内的 Line 79 (cv2.resize)
  
3. 修改 **测试数据文件夹** ：
    > **./eval_enh.py** 内的 test_dir 变量 (Line 100)  

4. 修改 **测试结果保存位置** ：
    > **./eval_enh.py** 内的 sv_dir 变量 (Line 129)

### 最后直接：python eval_enh.py

若需要测试 视频：  
1. 先运行 VideoGetFrame.py 获取 逐帧图像  
2. 运行上述测试环节获取增强图像
3. 运行 VideoCombineFrame.py 组合视频  
NOTICE: 注意文件夹位置及命名