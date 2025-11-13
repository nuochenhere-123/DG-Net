import os
os.environ['TORCH_HOME'] = './weights'
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.DG_Net import DGNet
from data.options_DG import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from loss.FFL_Loss import FocalFrequencyLoss as FFL
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
# NIQE
from loss.niqe_utils import *

opt = option().parse_args()

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def checkpoint(epoch):
    if not os.path.exists("./weights_DG"):          
        os.mkdir("./weights_DG") 
    if not os.path.exists("./weights_G/train_DG"):          
        os.mkdir("./weights_DG/train_DG")  
    model_out_path = "./weights_DG/train_DG/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    if opt.UIE:
            # # 正常读取 im1, im1_wb, im1_gc, im1_his, im2, file1, file1_wb, file1_gc, file1_his, file2
            # train_set = get_UIE_training_set(opt.data_train_UIE,size=opt.cropSize)
        # Step1 读取带无参考数据的输入 FR+NR
            train_set = get_UIE_training_set_with_NR(opt.data_train_UIE,size=opt.cropSize)
        # # Step1 读取带噪声项训练集的输入
        #     train_set = get_UIE_training_set_withNoise(opt.data_train_UIE,size=opt.cropSize)
        # # Step1_NR 读取带噪声项训练集的输入，无参考
        #     train_set = get_UIE_training_set_withNoise_NR(opt.data_train_UIE,size=opt.cropSize)

            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)

            # 正常读取验证集  input, input_wb, input_gc, input_his, file, file_wb, file_gc, file_his
            test_set = get_eval_set(opt.data_val_UIE)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)       
    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = DGNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights_DG/train_DG/epoch_{opt.start_epoch}.pth"
        print(f"load: ./weights_DG/train_DG/epoch_{opt.start_epoch}.pth")
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    loss_MSE = 0
    loss_MSE_raw = 0
    loss_Edge = 0
    loss_Edge_raw = 0
    loss_SSIM = 0
    loss_color = 0
    loss_colorVariance = 0
    loss_his = 0
    loss_light = 0
    loss_Vgg = 0
    loss_FFL = 0
    loss_Vgg_raw = 0
    loss_FFL_raw = 0
    loss_NIQE = 0
    loss_c = 0
    loss_c_L = 0
    pic_cnt = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        # # 正常读取
        # im1, im1_wb, im1_gc, im1_his, im2, path1, path1_wb, path1_gc, path1_his, path2 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9]
    # Step2 读取带无参考的输入
        im1, im1_wb, im1_gc, im1_his, im2, path1, path1_wb, path1_gc, path1_his, path2, im_NR, im_NR_wb, im_NR_gc, im_NR_his = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13]
        im_NR = im_NR.cuda()
        im_NR_wb = im_NR_wb.cuda()
        im_NR_gc = im_NR_gc.cuda()
        im_NR_his = im_NR_his.cuda()
        # im_NR_cp_his = im_NR_cp_his.cuda()
    # # Step2 读取带噪声项的输入
    #     im1, im1_wb, im1_gc, im1_his, im2, path1, path1_wb, path1_gc, path1_his, path2, im1_noise, im1_noise_wb, im1_noise_gc, im1_noise_his = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13]
    #     im1_noise = im1_noise.cuda()
    #     im1_noise_wb = im1_noise_wb.cuda()
    #     im1_noise_gc = im1_noise_gc.cuda()
    #     im1_noise_his = im1_noise_his.cuda()
    # # Step2_NR 读取带噪声项训练集的输入，无参考
    #     im1, im1_wb, im1_gc, im1_his, im2, im_NR, im_NR_wb, im_NR_gc, im_NR_his, im_NR_noise, im_NR_noise_wb, im_NR_noise_gc, im_NR_noise_his = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12]
    #     im_NR = im_NR.cuda()
    #     im_NR_wb = im_NR_wb.cuda()
    #     im_NR_gc = im_NR_gc.cuda()
    #     im_NR_his = im_NR_his.cuda()
    #     im_NR_noise = im_NR_noise.cuda()
    #     im_NR_noise_wb = im_NR_noise_wb.cuda()
    #     im_NR_noise_gc = im_NR_noise_gc.cuda()
    #     im_NR_noise_his = im_NR_noise_his.cuda()

        im1 = im1.cuda()
        im1_wb = im1_wb.cuda()
        im1_gc = im1_gc.cuda()
        im1_his = im1_his.cuda()
        im2 = im2.cuda()
        output_rgb, illumation_out, output_raw = model(im1, im1_wb, im1_gc, im1_his)  
    # Step3 输出无参考结果 NR
        output_rgb_NR, illumation_out_NR, output_raw_NR = model(im_NR, im_NR_wb, im_NR_gc, im_NR_his)  
    # # Step3 输出噪声项结果
    #     output_rgb_noise, illumation_out_noise = model(im1_noise, im1_noise_wb, im1_noise_gc, im1_noise_his)  
    # # Step3_NR 输出噪声项结果，无参考
    #     output_rgb_NR, illumation_out_NR = model(im_NR, im_NR_wb, im_NR_gc, im_NR_his)  
    #     output_rgb_NR_noise, illumation_out_NR_noise = model(im_NR_noise, im_NR_noise_wb, im_NR_noise_gc, im_NR_noise_his)  

        gt_rgb = im2
        # 有监督损失 加权重后的损失
        MSE = MSE_loss(output_rgb*255.0, gt_rgb*255.0)
        # Edge = E_loss(output_rgb*255.0, gt_rgb*255.0)
        # SSIM_D = D_loss(output_rgb, gt_rgb)
        # color_out = color_loss(output_rgb)
        Vgg = opt.P_weight * P_loss(output_rgb*255.0, gt_rgb*255.0)[0]
        FFL = opt.FFL_weight * FFL_loss(output_rgb*255.0, gt_rgb*255.0)

        MSE_raw = opt.MSE_raw_weight * MSE_loss(output_raw*255.0, im1*255.0)
        # Edge_raw = opt.MSE_raw_weight * E_loss(output_raw*255.0, im1*255.0)
        Vgg_raw = opt.MSE_raw_weight * opt.P_weight * P_loss(output_raw*255.0, im1*255.0)[0]
        FFL_raw = opt.MSE_raw_weight * opt.FFL_weight *  FFL_loss(output_raw*255.0, im1*255.0)

        # 无监督损失
        # 灰度世界：颜色均值一致性损失
        color_out = color_loss(output_rgb_NR*255.0)
        # 多样化色彩：颜色方差损失
        color_var_out = colorVariance_loss(output_rgb_NR*255.0)
        # # 直方图分布一致性损失
        # his_loss = histogram_loss(output_rgb_NR*255.0, im_NR_cp_his*255.0)

    # # Step4 无监督损失，一致性正则
    #     consisency_N_out = opt.consistency_weight * MSE_loss(output_rgb_noise*255, output_rgb*255)
    #     consisency_N_L = opt.consistency_weight_L * MSE_loss(illumation_out_noise*255, illumation_out*255)
    # # Step4_NR 无监督损失
    #     # consisency_N_out = opt.consistency_weight * MSE_loss(output_rgb_NR_noise*255, output_rgb_NR*255) 
    #     # consisency_N_L = opt.consistency_weight_L * MSE_loss(illumation_out_NR_noise*255, illumation_out_NR*255)
    #     MSE = MSE_loss(output_rgb*255, im1_wb*255)
    #     Edge = E_loss(output_rgb, im1)
    #     # SSIM_D = D_loss(output_rgb, im1)
    #     color_out = color_loss(output_rgb)
    #     colorVariance_out = colorVariance_loss(output_rgb)
    #     light_out = light_Loss(output_rgb)
    #     Vgg = opt.P_weight * P_loss(output_rgb*255, im1_wb*255)[0]
    #     # FFL = opt.FFL_weight * FFL_loss(output_rgb*255, im1*255)

        # # NIQE
        # niqe_loss_total = 0.0
        # niqe_target = 2.0   # 定义一个目标范围，例如更自然图像 score 趋于 3.0（低分更好）
        # for i in range(output_rgb.shape[0]):
        #     out_np = output_rgb[i].detach().cpu().numpy().transpose(1, 2, 0)
        #     out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        #     score = calculate_niqe(out_np, input_order='HWC', convert_to='y')
        #     niqe_loss_total += (score - niqe_target) ** 2  # 目标分越小越自然

        # niqe_guidance_loss = torch.tensor(opt.NIQE_weight * niqe_loss_total/output_rgb.shape[0]).cuda()


        # # 总损失 阶段性动态权重
        # if epoch <= opt.nEpochs/2:
        #     loss = MSE + Edge + color_out + light_out + Vgg # MSE + 10000Edge + 600 color + 600 light + 5e-4 Vgg
        # else:
        #     loss = Edge + color_out + light_out + Vgg # 10000Edge + 600 color + 600 light + 5e-4 Vgg
        # 正常损失
        # loss = MSE + MSE_raw + Vgg + Vgg_raw + FFL + FFL_raw + color_out - color_var_out + his_loss # 惩罚小方差，即鼓励方差大
        loss = MSE + MSE_raw + Vgg + Vgg_raw + FFL + FFL_raw + color_out - color_var_out # 惩罚小方差，即鼓励方差大
        # loss = MSE + Edge + color_out + colorVariance_out + light_out + Vgg + niqe_guidance_loss 
    # # Step5 一致性正则 / 无监督损失
    #     loss = MSE + Vgg + FFL + consisency_N_out + consisency_N_L
        # loss = MSE + Vgg + FFL + consisency_N_L
        
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2) # 启用梯度裁剪，用于防止梯度爆炸，使用 L2 范数
        
        optimizer.zero_grad() # 清空上一步的梯度
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 用计算出的梯度更新模型参数
        
        loss_print = loss_print + loss.item()
        loss_MSE = loss_MSE + MSE.item()
        loss_MSE_raw = loss_MSE_raw + MSE_raw.item()
        # loss_Edge = loss_Edge + Edge.item()
        # loss_Edge_raw = loss_Edge_raw + Edge_raw.item()
        # loss_SSIM = loss_SSIM + SSIM_D.item()
        # loss_light = loss_light + light_out.item()
        loss_Vgg = loss_Vgg + Vgg.item()
        loss_Vgg_raw = loss_Vgg_raw + Vgg_raw.item()
        # loss_NIQE = loss_NIQE + niqe_guidance_loss.item()
        loss_FFL = loss_FFL + FFL.item()
        loss_FFL_raw = loss_FFL_raw + FFL_raw.item()

        loss_color = loss_color + color_out.item()
        loss_colorVariance = loss_colorVariance + color_var_out.item()
        # loss_his = loss_his + his_loss.item()
        # # # 正常时无正则化项
        # loss_c = 0
        # loss_c_L = 0
        loss_his = 0
    # # Step6 记录
    #     loss_c = loss_c + consisency_N_out.item()
    #     loss_c_L = loss_c_L + consisency_N_L.item()

        pic_cnt += 1
        if iter == train_len:
            # print("===> Epoch[{}]: Avg_Loss: {:.4f} Edge: {:.4f} color_loss: {:.4f} colorVariance_loss: {:.4f} Light: {:.4f} Vgg: {:.4f} MSE: {:.4f} NIQE: {:.4f}  || Learning rate: lr={}.".format(epoch,
            #     loss_print/pic_cnt, loss_Edge/pic_cnt, loss_color/pic_cnt, loss_colorVariance/pic_cnt, loss_light/pic_cnt, loss_Vgg/pic_cnt, loss_MSE/pic_cnt, loss_NIQE/pic_cnt, optimizer.param_groups[0]['lr'])) # 输出当前 epoch 的平均损失和当前学习率
            print("===> Epoch[{}]: Avg_Loss: {:.4f} MSE: {:.4f} MSE_raw: {:.4f} VGG: {:.4f} VGG_raw: {:.4f} FFL: {:.4f} FFL_raw: {:.4f}\n color: {:.4f} color_var: {:.4f} HistogramEMDLoss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_print/pic_cnt, loss_MSE/pic_cnt, loss_MSE_raw/pic_cnt, loss_Vgg/pic_cnt, loss_Vgg_raw/pic_cnt, loss_FFL/pic_cnt, loss_FFL_raw/pic_cnt, loss_color/pic_cnt, loss_colorVariance/pic_cnt, loss_his/pic_cnt, optimizer.param_groups[0]['lr'])) # 输出当前 epoch 的平均损失和当前学习率
            
            # 保存 全参考 结果
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0)) # (output_rgb)[0]：选出 batch 中第一个样本  .squeeze(0)：去掉 batch 维度
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/res.png')
            gt_img.save(opt.val_folder+'training/gt.png')

            # 保存 无监督 结果
            output_rgb_NR_img = transforms.ToPILImage()((output_rgb_NR)[0].squeeze(0)) # ()[0]：选出 batch 中第一个样本  .squeeze(0)：去掉 batch 维度
            # im_NR_cp_his_img = transforms.ToPILImage()((im_NR_cp_his)[0].squeeze(0))
            output_rgb_NR_img.save(opt.val_folder+'training/res_NR.png')
            # im_NR_cp_his_img.save(opt.val_folder+'training/res_NR_HisAfterCP.png')

    # return loss_print, pic_cnt, optimizer.param_groups[0]['lr'], loss_MSE/pic_cnt, loss_Edge/pic_cnt, loss_color/pic_cnt, loss_colorVariance/pic_cnt, loss_Vgg/pic_cnt, loss_light/pic_cnt, loss_NIQE/pic_cnt
    return loss_print, pic_cnt, optimizer.param_groups[0]['lr'], loss_MSE/pic_cnt, loss_MSE_raw/pic_cnt, loss_Vgg/pic_cnt, loss_Vgg_raw/pic_cnt, loss_FFL/pic_cnt, loss_FFL_raw/pic_cnt, loss_color/pic_cnt, loss_colorVariance/pic_cnt, loss_his/pic_cnt
                

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic: # 周期性余弦退火调度器
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart: # 余弦退火调度器
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            # 有多个训练阶段，也可以设置多个周期
            # 如 periods = [30, 30, 40]  # 三次重启，总共 100 个 epoch
            # restart_weights = [1, 0.5, 0.25] # 每次重启时学习率的缩放比例
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    MSE_weight   = opt.MSE_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    color_weight = opt.color_weight
    colorStd_weight = opt.colorStd_weight  
    histogram_weight = opt.histgram_weight  
    Light_weight = opt.lightness_weight
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    MSE_loss= MSELoss(loss_weight=MSE_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    # 灰度世界
    color_loss = ColorBalanceLoss(loss_weight=color_weight, reduction='mean').cuda()
    # 颜色多样性
    colorVariance_loss = ColorVarianceLoss(loss_weight=colorStd_weight, reduction='mean').cuda()
    # 直方图分布差异
    histogram_loss = HistogramEMDLoss(loss_weight=histogram_weight, bins=64, reduction='mean').cuda()

    light_Loss = LightnessLoss(target_value=0.6, loss_weight=Light_weight, reduction='mean').cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    FFL_loss = FFL(loss_weight=1.0, alpha=1.0).cuda()

    return L1_loss,P_loss,E_loss,D_loss,MSE_loss,FFL_loss,color_loss,light_Loss,colorVariance_loss,histogram_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    # training_data_loader: im1, im1_wb, im1_gc, im1_his, im2, file1, file1_wb, file1_gc, file1_his, file2
    training_data_loader, testing_data_loader = load_datasets() 
    # testing_data_loader: input, input_wb, input_gc, input_his, file, file_wb, file_gc, file_his
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss,MSE_loss,FFL_loss,color_loss,light_Loss,colorVariance_loss,histogram_loss = init_loss()
    
    '''
    train
    '''
    psnr, ssim, lpips, uiqm, uciqe = [], [], [], [], []
    now_loss, LR_now, MSE_now, MSE_raw_now, Edge_now, Edge_raw_now, SSIM_now, VGG_now, VGG_raw_now, FFL_now, FFL_raw_now, c_now, c_L_now, color_now, colorVariance_now, his_now, light_now, niqe_now = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + 1):
        print("Start epoch: ", epoch, " of total ", opt.nEpochs, "epoches.")
        # epoch_loss, pic_num, LR_epoch, MSE_epoch, Edge_epoch, color_epoch, colorVariance_epoch, Vgg_epoch, light_epoch, niqe_epoch = train(epoch)
        epoch_loss, pic_num, LR_epoch, MSE_epoch, MSE_raw_epoch, Vgg_epoch, Vgg_raw_epoch, FFL_epoch, FFL_raw_epoch, color_epoch, color_var_epoch, his_epoch = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0 or epoch == opt.nEpochs:
            model_out_path = checkpoint(epoch)  # 保存当前epoch权重并输出保存路径
            norm_size = True

            if opt.UIE:
                output_folder = 'val-60/' # 验证集训练过程中输出 文件夹 命名
                label_dir = opt.data_valgt_UIE # reference-60
            
            im_dir = opt.val_folder + output_folder + '*.png' # 验证集训练过程中输出 文件夹
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, # 验证集数据，该epoch权重位置，验证集输出结果文件夹
                 LOL=opt.UIE)
            avg_psnr, avg_ssim, avg_lpips, avg_uiqm, avg_uciqe = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.5f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.5f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.5f} ".format(avg_lpips))
            print("===> Avg.UIQM: {:.5f} ".format(avg_uiqm))
            print("===> Avg.UCIQE: {:.5f} ".format(avg_uciqe))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            uiqm.append(avg_uiqm)
            uciqe.append(avg_uciqe)
            now_loss.append(epoch_loss/pic_num)
            LR_now.append(LR_epoch)
            MSE_now.append(MSE_epoch)
            MSE_raw_now.append(MSE_raw_epoch)
            # Edge_now.append(Edge_epoch)
            # Edge_raw_now.append(Edge_raw_epoch)
            # SSIM_now.append(SSIM_epoch)
            # light_now.append(light_epoch)
            # niqe_now.append(niqe_epoch)
            VGG_now.append(Vgg_epoch)
            VGG_raw_now.append(Vgg_raw_epoch)
            FFL_now.append(FFL_epoch)
            FFL_raw_now.append(FFL_raw_epoch)
            color_now.append(color_epoch)
            colorVariance_now.append(color_var_epoch)
            his_now.append(his_epoch)
            # c_now.append(c_epoch)
            # c_L_now.append(c_L_epoch)
            print("PSNR: ", psnr)
            print("SSIM: ", ssim)
            print("LPIPS: ", lpips)
            print("UIQM: ", uiqm)
            print("UCIQE: ", uciqe)
            print("LR: ", LR_epoch)
            print("MSE: ", MSE_now)
            print("MSE_raw: ", MSE_raw_now)
            # print("Edge: ", Edge_now)
            # print("Edge_raw: ", Edge_raw_now)
            # print("SSIM: ", SSIM_now)
            # print("Light: ", light_now)
            # print("NIQE: ", niqe_now)
            print("VGG: ",VGG_now)
            print("VGG_raw: ",VGG_raw_now)
            print("FFL: ",FFL_now)
            print("FFL_raw: ",FFL_raw_now)
            print("Color: ", color_now)
            print("ColorVariance: ", colorVariance_now)
            print("HistogramEMDLoss: ", his_now)

        # # Step7 打印
        #     print("consisency_N_out: ",c_now)
        #     print("consisency_N_L: ",c_L_now)
            print("Final_Loss: ", now_loss)
            # print("Edge: ", Edge_epoch, "Color: ", color_epoch, "ColorVariance: ", colorVariance_epoch, "Light: ", light_epoch, "VGG: ", Vgg_epoch, "MSE: ", MSE_epoch, "NIQE: ", niqe_epoch)
            print("MSE: ", MSE_epoch, "MSE_raw: ", MSE_raw_epoch, "VGG: ", Vgg_epoch, "VGG_raw: ", Vgg_raw_epoch, "FFL: ", FFL_epoch, "FFL_raw: ", FFL_raw_epoch, "\ncolor: ", color_epoch, "color_var: ", color_var_epoch, "HistogramEMDLoss: ", his_epoch )
        torch.cuda.empty_cache()
    
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./{opt.val_folder}/training/metrics_record_{now}.md", "w") as f:
        f.write("dataset: " + output_folder + "\n")  
        f.write(f"lr_start: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"MSE_weight: {opt.MSE_weight}\n")  
        f.write(f"MSE_raw_weight: {opt.MSE_raw_weight}\n")  
        # f.write(f"Edge_weight: {opt.E_weight}\n")  
        # f.write(f"Edge_raw_weight: {opt.MSE_raw_weight}\n")  
        # f.write(f"SSIM_weight: {opt.D_weight}\n")  
        # f.write(f"light_weight: {opt.lightness_weight}\n")  
        # f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"VGG_weight: {opt.P_weight}\n")  
        f.write(f"VGG_raw_weight: {opt.MSE_raw_weight}\n")  
        # f.write(f"NIQE_weight: {opt.NIQE_weight}\n")  
        f.write(f"FFL_weight: {opt.FFL_weight}\n")         
        f.write(f"FFL_raw_weight: {opt.MSE_raw_weight}\n")   
        f.write(f"color_weight: {opt.color_weight}\n")  
        f.write(f"colorVariance_weight: {opt.colorStd_weight}   R: 1000-6000, G: 1000-5500, B: 1000-5900 \n")  
        # f.write(f"HistogramEMDLoss_weight: {opt.histgram_weight}\n")
        # f.write(f"consistency_out_weight: {opt.consistency_weight}\n")  
        # f.write(f"consistency_weight_L: {opt.consistency_weight_L}\n")  
        # f.write(f"\n LOSS: {opt.MSE_weight} MSE + {opt.E_weight} Edge + {opt.color_weight} color + {opt.colorStd_weight} colorVariance + {opt.lightness_weight} light + {opt.P_weight} Vgg + {opt.NIQE_weight} NIQE\n")
        # f.write(f"\n LOSS: {opt.MSE_weight} MSE + {opt.MSE_weight * opt.MSE_raw_weight} MSE_raw + {opt.P_weight} Vgg + {opt.P_weight * opt.MSE_raw_weight} Vgg_raw + {opt.FFL_weight} FFL + {opt.FFL_weight * opt.MSE_raw_weight} FFL_raw + {opt.color_weight} color + {opt.colorStd_weight} color_var + {opt.histgram_weight} HistogramEMDLoss\n")
        f.write(f"\n LOSS: {opt.MSE_weight} MSE + {opt.MSE_weight * opt.MSE_raw_weight} MSE_raw + {opt.P_weight} Vgg + {opt.P_weight * opt.MSE_raw_weight} Vgg_raw + {opt.FFL_weight} FFL + {opt.FFL_weight * opt.MSE_raw_weight} FFL_raw + {opt.color_weight} color + {opt.colorStd_weight} color_var\n")
        f.write("|  Epochs  |  PSNR  |  SSIM  |  LPIPS  |   UIQM   |  UCIQE  |    LR    |   Loss  |   MSE   |   MSE_raw   |   VGG   |   VGG_raw   |   FFL   |   FFL_raw   |   color   |   color_var   | HistogramEMDLoss |\n")  
        f.write("|----------|--------|--------|---------|----------|---------|----------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i]:.5f} | {ssim[i]:.5f} | {lpips[i]:.5f} | {uiqm[i]:.5f} | {uciqe[i]:.5f} | {LR_now[i]:.7f} | {now_loss[i]:.5f} |")
            f.write(f" {MSE_now[i]:.5f} | {MSE_raw_now[i]:.5f} | {VGG_now[i]:.5f} | {VGG_raw_now[i]:.5f} | {FFL_now[i]:.5f} | {FFL_raw_now[i]:.5f} | {color_now[i]:.5f} | {colorVariance_now[i]:.5f} | {his_now[i]:.5f} |\n")

# cd RCG-Net
# python train_DGNet.py
