import argparse

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=5, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=20, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=10, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # train datasets 母文件夹，子文件夹定义见 data.py 引用
    parser.add_argument('--data_train_UIE'     , type=str, default='./datasets/780')
    
    # validation input
    parser.add_argument('--data_val_UIE'     , type=str, default='./datasets/60')
   
    # validation groundtruth
    parser.add_argument('--data_valgt_UIE'     , type=str, default='./datasets/60/reference-60/')
    
    parser.add_argument('--val_folder', default='./results_Cycle_FR&NR/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0) # 定义里返回
    parser.add_argument('--MSE_weight', type=float, default=1.0) # 定义里返回               
    parser.add_argument('--MSE_raw_weight', type=float, default=0.1) # train内相乘              
    parser.add_argument('--D_weight',  type=float, default=1.5) # 定义里返回 SSIM
    parser.add_argument('--E_weight',  type=float, default=0.0025) # 定义里返回 Edge      
    parser.add_argument('--color_weight',  type=float, default=0.02) # 定义里返回 color  灰度世界     
    parser.add_argument('--colorStd_weight',  type=float, default=0.004) # 定义里返回 color   色彩多样性 利用惩罚项控制在1200-4300  
    parser.add_argument('--histgram_weight',  type=float, default=0.05) # 定义里返回 histogram   直方图分布差异  
    parser.add_argument('--lightness_weight',  type=float, default=80) # 定义里返回 color   
    parser.add_argument('--P_weight',  type=float, default=5e-6) # train内5-++-*-相乘 VGG   
    parser.add_argument('--FFL_weight', type=float, default=12e-4) # train内相乘
    parser.add_argument('--NIQE_weight', type=float, default=1.2) # train内相乘             
    parser.add_argument('--consistency_weight', type=float, default=0.2) # 噪声结果的一致性正则化项
    parser.add_argument('--consistency_weight_L', type=float, default=0.1) # 噪声结果的反射率一致性正则化项
    
    # # use random gamma function (enhancement curve) to improve generalization
    # parser.add_argument('--gamma', type=bool, default=False)
    # parser.add_argument('--start_gamma', type=int, default=60)
    # parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    
    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--UIE', type=bool, default=True)

    return parser
