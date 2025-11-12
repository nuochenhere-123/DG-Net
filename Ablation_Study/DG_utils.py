import torch
import torch.nn as nn
import torch.nn.functional as F
from net.RCG_Attention import *

# RGB 空间 反射率估计
# LAB 空间 照度估计
class encoderRGBLAB(nn.Module):
    def __init__(self, channels=[32, 64, 128], bias=True):
        super(encoderRGBLAB, self).__init__()

        [out_channel1, out_channel2, out_channel3] = channels

        # 第一层 encoder 
        # 多尺度前细化
        self.convBDBM1 = nn.Sequential(
            nn.Conv2d(12, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.ReLU()
        )
        self.detailBM1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导1
        self.convBDBM1_DQ1_A = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=1, stride=1, padding=0, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_B = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_C = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=5, stride=1, padding=2, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.SCA_DQ1 =  RCG_SCA((out_channel1//2)*3, ratio=1, bias=True)
        self.convBDAM1_DQ1_all = nn.Sequential(
            nn.Conv2d((out_channel1//2)*3, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM1_DQ1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导1
        # 多尺度卷积
        self.multiConv11 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU()
        )
        self.multiConv31 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        self.multiConv51 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.ReLU()
        )
        self.SCA1 =  RCG_SCA(out_channel1*3, ratio=1, bias=True)
        # 多尺度后细化
        self.convBDAM1 = nn.Sequential(
            nn.Conv2d(out_channel1*3, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.ReLU()
        )
        self.detailAM1 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 第一层 encoder
    # DQ 引导2
        self.convBDBM1_DQ1_A_1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=1, stride=1, padding=0, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_B_1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_C_1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=5, stride=1, padding=2, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.SCA_DQ2 =  RCG_SCA(out_channel1*3, ratio=1, bias=True)
        self.convBDAM1_DQ1_all_1 = nn.Sequential(
            nn.Conv2d(out_channel1*3, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM1_DQ1_1 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导2

         
        # 第二层 encoder 
        # 多尺度前细化
        self.convBDBM2 = nn.Sequential(
            nn.Conv2d(12, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.ReLU()
        )
        self.detailBM2 = nn.Sequential(
            nn.Conv2d(out_channel2//2, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2//2, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2//2, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 多尺度卷积
        self.multiConv12 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU()
        )
        self.multiConv32 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        self.multiConv52 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.ReLU()
        )
        self.SCA2 =  RCG_SCA(out_channel2*3, ratio=1, bias=True)
        # 多尺度后细化
        self.convBDAM2 = nn.Sequential(
            nn.Conv2d(out_channel2*3, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.ReLU()
        )
        self.detailAM2 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 第二层 encoder
    # DQ 引导3
        self.convBDBM2_DQ2_A = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=1, stride=1, padding=0, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM2_DQ2_B = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM2_DQ2_C = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=5, stride=1, padding=2, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.SCA_DQ3 =  RCG_SCA(out_channel2*3, ratio=1, bias=True)
        self.convBDAM2_DQ2_all = nn.Sequential(
            nn.Conv2d(out_channel2*3, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM2_DQ2 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导3

        # 第三层 encoder
        # 多尺度前细化
        self.convBDBM3 = nn.Sequential(
            nn.Conv2d(12, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.ReLU()
        )
        self.detailBM3 = nn.Sequential(
            nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 多尺度卷积
        self.multiConv13 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU()
        )
        self.multiConv33 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        self.multiConv53 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.ReLU()
        )
        self.SCA3 =  RCG_SCA(out_channel3*3, ratio=1, bias=True)
        # 多尺度后细化
        self.convBDAM3 = nn.Sequential(
            nn.Conv2d(out_channel3*3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.ReLU()
        )
        self.detailAM3 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 第三层 encoder


        # # 细化
        # self.SCA_RGB =  RCG_SCA(out_channel3, ratio=1, bias=True)
        # self.convBDAM_RGB = nn.Sequential(
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
        #     nn.ReLU()
        # )
        # self.detailAM_RGB = nn.Sequential(
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias)
        # )

        # # ConfidenceGuided
        # self.confidenceGuided = nn.Sequential(
        #     nn.Conv2d(12, out_channel3, kernel_size=7, stride=1, padding=3, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=5, stride=1, padding=2, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3//2, kernel_size=1, stride=1, padding=0, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=7, stride=1, padding=3, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=5, stride=1, padding=2, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, 3, kernel_size=3, stride=1, padding=1, bias=bias), 
        #     nn.Sigmoid()
        # )


        # Decoder 第一层
        # 上采样 + 注意力
        # self.convUp1 = nn.Sequential(
        #     nn.Conv2d(out_channel3, out_channel3*4, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel3*4, out_channel3*4, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.PixelShuffle(upscale_factor=2)  # 通道数 512/4=128 , HxW -> 2Hx2W
        # )
        self.SCA_up1 = RCG_SCA(out_channel3+out_channel2, ratio=1, bias=True)
        # 细化
        self.convUpToRefine11 = nn.Sequential(
            nn.Conv2d(out_channel3+out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        # self.detailUp11 = nn.Sequential(
        #     nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        # )
        self.convUpToRefine12 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        self.detailUp12 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        )


        # Decoder 第二层
        # 上采样 + 注意力
        self.convUp2 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel2*4, out_channel2*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(upscale_factor=2)  # 通道数 512/4=128 , HxW -> 2Hx2W
        )
        self.SCA_up2 = RCG_SCA(out_channel2+out_channel1, ratio=1, bias=True)
        # 细化
        self.convUpToRefine21 = nn.Sequential(
            nn.Conv2d(out_channel2+out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        # self.detailUp21 = nn.Sequential(
        #     nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        # )
        self.convUpToRefine22 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        self.detailUp22 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        )


        # Decoder 第三层
        # 上采样 + 注意力
        self.convUp3 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1*4, out_channel1*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(upscale_factor=2)  # 通道数 512/4=128 , HxW -> 2Hx2W
        )
        self.SCA_up3 = RCG_SCA(out_channel1+out_channel1//2, ratio=1, bias=True)
        # 细化
        self.convUpToRefine31 = nn.Sequential(
            nn.Conv2d(out_channel1+out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        # self.detailUp31 = nn.Sequential(
        #     nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        # )
        self.convUpToRefine32 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU()
        )
        self.detailUp32 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        # 该分支最终输出
        self.outConv = nn.Sequential(
            nn.Conv2d(out_channel1//2, 3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Sigmoid(),
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, im1, im1_wb, im1_gc, im1_his):
        # 第一层 encoder   # 缺一个concat输入 inputX
        # 多尺度前细化
        ToRefine = self.convBDBM1(torch.cat([im1, im1_wb, im1_gc, im1_his], dim=1))
        Refined = self.detailBM1(ToRefine)
        Refined_res1 = ToRefine + Refined
    # DQ1
        DQ1_A = self.convBDBM1_DQ1_A(torch.cat([im1_wb-im1, im1_gc-im1, im1_his-im1], dim=1))
        DQ1_B = self.convBDBM1_DQ1_B(torch.cat([im1_gc-im1, im1_his-im1, im1_wb-im1], dim=1))
        DQ1_C = self.convBDBM1_DQ1_C(torch.cat([im1_his-im1, im1_wb-im1, im1_gc-im1], dim=1))
        DQ1_SCA = self.SCA_DQ1(torch.cat([DQ1_A, DQ1_B, DQ1_C], dim=1))
        DQ1_all = self.convBDAM1_DQ1_all(DQ1_SCA)
        DQ1_Refined = self.detailBM1_DQ1(DQ1_all)
        DQ1_Refined = DQ1_all + DQ1_Refined
    # DQ1
        # 多尺度卷积
        x1 = self.multiConv11(Refined_res1)
        x3 = self.multiConv31(Refined_res1)
        x5 = self.multiConv51(Refined_res1)
        mt_SCA = self.SCA1(torch.cat([x1, x3, x5], dim=1))
        # 多尺度后细化
        ToRefine = self.convBDAM1(mt_SCA)
        Refined = self.detailAM1(ToRefine)
        Refined = ToRefine + Refined
        # 最大池化
        res1 = self.pool(Refined)
    # DQ2
        DQ1_Refined1 = self.pool(DQ1_Refined)
        DQ1_A_1 = self.convBDBM1_DQ1_A_1(DQ1_Refined1)
        DQ1_B_1 = self.convBDBM1_DQ1_B_1(DQ1_Refined1)
        DQ1_C_1 = self.convBDBM1_DQ1_C_1(DQ1_Refined1)
        DQ1_1_SCA = self.SCA_DQ2(torch.cat([DQ1_A_1, DQ1_B_1, DQ1_C_1], dim=1))
        DQ1_all_1 = self.convBDAM1_DQ1_all_1(DQ1_1_SCA)
        DQ1 = self.detailBM1_DQ1_1(DQ1_all_1)
        DQ1 = DQ1_all_1 + DQ1
    # DQ2
        

        # 第二层 encoder
        scale2 = self.pool(torch.cat([im1, im1_wb, im1_gc, im1_his], dim=1))
        # 多尺度前细化
        ToRefine = self.convBDBM2(scale2)
        Refined = self.detailBM2(ToRefine)
        Refined = ToRefine + Refined
        # 多尺度卷积
        x1 = self.multiConv12(torch.cat([res1, Refined], dim=1))
        x3 = self.multiConv32(torch.cat([res1, Refined], dim=1))
        x5 = self.multiConv52(torch.cat([res1, Refined], dim=1))
        mt_SCA = self.SCA2(torch.cat([x1, x3, x5], dim=1))
        # 多尺度后细化
        ToRefine = self.convBDAM2(mt_SCA)
        Refined = self.detailAM2(ToRefine)
        Refined = ToRefine + Refined
        # 最大池化
        res2 = self.pool(Refined)
    # DQ3
        DQ1_1 = self.pool(DQ1)
        DQ2_A = self.convBDBM2_DQ2_A(DQ1_1)
        DQ2_B = self.convBDBM2_DQ2_B(DQ1_1)
        DQ2_C = self.convBDBM2_DQ2_C(DQ1_1)
        DQ2_SCA = self.SCA_DQ3(torch.cat([DQ2_A, DQ2_B, DQ2_C], dim=1))
        DQ2_all = self.convBDAM2_DQ2_all(DQ2_SCA)
        DQ2 = self.detailBM2_DQ2(DQ2_all)
        DQ2 = DQ2_all + DQ2
    # DQ3


        # 第三层 encoder
        scale3 = self.pool(scale2)
        # 多尺度前细化
        ToRefine = self.convBDBM3(scale3)
        Refined = self.detailBM3(ToRefine)
        Refined = ToRefine + Refined
        # 多尺度卷积
        x1 = self.multiConv13(torch.cat([res2, Refined], dim=1))
        x3 = self.multiConv33(torch.cat([res2, Refined], dim=1))
        x5 = self.multiConv53(torch.cat([res2, Refined], dim=1))
        mt_SCA = self.SCA3(torch.cat([x1, x3, x5], dim=1))
        # 多尺度后细化
        ToRefine = self.convBDAM3(mt_SCA)
        Refined = self.detailAM3(ToRefine)
        # Refined = ToRefine + Refined
        res3 = ToRefine + Refined
        # # 最大池化
        # res3 = self.pool(Refined)


        # # 细化
        # SCA_RGB = self.SCA_RGB(res3)
        # ToRefine = self.convBDAM_RGB(SCA_RGB)
        # Refined = self.convBDAM_RGB(ToRefine)
        # Fusion_Refined = ToRefine + Refined

        # # ConfidenceGuided
        # weight = self.pool(scale3)
        # weight = self.confidenceGuided(weight)
        # out_wb, out_gc, out_histeq = torch.split(weight, 1, dim=1)  # 沿着通道维度拆分为三个单通道
        # res_CG = Fusion_Refined * out_wb + Fusion_Refined * out_gc + Fusion_Refined * out_histeq


        # Decoder 第一层
        # up1 = self.convUp1(Fusion_Refined)
        # print("up1 shape:", up1.shape)
        # print("res2 shape:", res2.shape)
        # 确保 up1 和 res2 尺寸一致
        if res3.size(2) != res2.size(2) or res3.size(3) != res2.size(3):
            res3 = F.interpolate(res3, size=(res2.size(2), res2.size(3)), mode='bilinear', align_corners=True)
        up1_SCA = self.SCA_up1(torch.cat([res3, res2], dim=1))   
    # DQ3
        # 细化
        up1ToRefine = self.convUpToRefine11(up1_SCA)
        # 确保 DQ2 和 up1ToRefine 尺寸一致
        if DQ2.size(2) != up1ToRefine.size(2) or DQ2.size(3) != up1ToRefine.size(3):
            DQ2 = F.interpolate(DQ2, size=(up1ToRefine.size(2), up1ToRefine.size(3)), mode='bilinear', align_corners=True)
        DQ2_res = DQ2 * up1ToRefine + up1ToRefine
    # DQ3
        # up1Refined = self.detailUp11(up1ToRefine)
        # up1Refined = up1ToRefine + up1Refined
        up1ToRefine = self.convUpToRefine12(DQ2_res)
        up1Refined = self.detailUp12(up1ToRefine)
        up1Refined = up1ToRefine + up1Refined


        # Decoder 第二层
        up2 = self.convUp2(up1Refined)
        # print("up2 shape:", up2.shape)
        # print("res1 shape:", res1.shape)
        # 确保 up2 和 res1 尺寸一致
        if up2.size(2) != res1.size(2) or up2.size(3) != res1.size(3):
            up2 = F.interpolate(up2, size=(res1.size(2), res1.size(3)), mode='bilinear', align_corners=True)
        up2_SCA = self.SCA_up2(torch.cat([up2, res1], dim=1))
    # DQ2
        up2ToRefine = self.convUpToRefine21(up2_SCA)
        # 确保 DQ1 和 up1ToRefine 尺寸一致
        if DQ1.size(2) != up2ToRefine.size(2) or DQ1.size(3) != up2ToRefine.size(3):
            DQ1 = F.interpolate(DQ1, size=(up2ToRefine.size(2), up2ToRefine.size(3)), mode='bilinear', align_corners=True)
        DQ1_res = DQ1 * up2ToRefine + up2ToRefine
    # DQ2
        up2ToRefine = self.convUpToRefine22(DQ1_res)
        up2Refined = self.detailUp22(up2ToRefine)
        up2Refined = up2ToRefine + up2Refined

        # Decoder 第三层
        up3 = self.convUp3(up2Refined)
        # 确保 up3 和 Refined_res1 尺寸一致
        if up3.size(2) != Refined_res1.size(2) or up3.size(3) != Refined_res1.size(3):
            up3 = F.interpolate(up3, size=(Refined_res1.size(2), Refined_res1.size(3)), mode='bilinear', align_corners=True)
        up3_SCA = self.SCA_up3(torch.cat([up3, Refined_res1], dim=1))
    # DQ1
        up3ToRefine = self.convUpToRefine31(up3_SCA)
        # 确保 DQ1_ 和 up1ToRefine 尺寸一致
        if DQ1_Refined.size(2) != up3ToRefine.size(2) or DQ1_Refined.size(3) != up3ToRefine.size(3):
            DQ1_Refined = F.interpolate(DQ1_Refined, size=(up3ToRefine.size(2), up3ToRefine.size(3)), mode='bilinear', align_corners=True)
        DQ1_Refined_res = DQ1_Refined * up3ToRefine + up3ToRefine
    # DQ1
        up3ToRefine = self.convUpToRefine32(DQ1_Refined_res)
        up3Refined = self.detailUp32(up3ToRefine)
        up3Refined = up3ToRefine + up3Refined
        
        # 该分支最终输出
        final_output = self.outConv(up3Refined)

        return final_output




# noise 空间 噪声估计
class encoderNoise(nn.Module):
    def __init__(self, channels=[32, 64, 128], bias=True):
        super(encoderNoise, self).__init__()

        [out_channel1, out_channel2, out_channel3] = channels

        # 第一层 encoder 
        # 多尺度前细化
        self.convBDBM1 = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导1
        self.convBDBM1_DQ1_A = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=1, stride=1, padding=0, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_B = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_C = nn.Sequential(
            nn.Conv2d(9, out_channel1//2, kernel_size=5, stride=1, padding=2, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.SCA_DQ1 =  RCG_SCA((out_channel1//2)*3, ratio=1, bias=True)
        self.convBDAM1_DQ1_all = nn.Sequential(
            nn.Conv2d((out_channel1//2)*3, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM1_DQ1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导1
        # 多尺度卷积
        self.multiConv11 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh()
        )
        self.multiConv31 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        self.multiConv51 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.Tanh()
        )
        self.SCA1 =  RCG_SCA(out_channel1*3, ratio=1, bias=True)
        # 多尺度后细化
        self.convBDAM1 = nn.Sequential(
            nn.Conv2d(out_channel1*3, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailAM1 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 第一层 encoder
    # DQ 引导2
        self.convBDBM1_DQ1_A_1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=1, stride=1, padding=0, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_B_1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM1_DQ1_C_1 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1, kernel_size=5, stride=1, padding=2, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.SCA_DQ2 =  RCG_SCA(out_channel1*3, ratio=1, bias=True)
        self.convBDAM1_DQ1_all_1 = nn.Sequential(
            nn.Conv2d(out_channel1*3, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM1_DQ1_1 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导2

         
        # 第二层 encoder 
        # 多尺度前细化
        self.convBDBM2 = nn.Sequential(
            nn.Conv2d(9, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM2 = nn.Sequential(
            nn.Conv2d(out_channel2//2, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2//2, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2//2, out_channel2//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 多尺度卷积
        self.multiConv12 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh()
        )
        self.multiConv32 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        self.multiConv52 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.Tanh()
        )
        self.SCA2 =  RCG_SCA(out_channel2*3, ratio=1, bias=True)
        # 多尺度后细化
        self.convBDAM2 = nn.Sequential(
            nn.Conv2d(out_channel2*3, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailAM2 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 第二层 encoder
    # DQ 引导3
        self.convBDBM2_DQ2_A = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=1, stride=1, padding=0, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM2_DQ2_B = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.convBDBM2_DQ2_C = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=5, stride=1, padding=2, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.SCA_DQ3 =  RCG_SCA(out_channel2*3, ratio=1, bias=True)
        self.convBDAM2_DQ2_all = nn.Sequential(
            nn.Conv2d(out_channel2*3, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM2_DQ2 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
    # DQ 引导3


        # 第三层 encoder
        # 多尺度前细化
        self.convBDBM3 = nn.Sequential(
            nn.Conv2d(9, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailBM3 = nn.Sequential(
            nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 多尺度卷积
        self.multiConv13 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh()
        )
        self.multiConv33 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        self.multiConv53 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=5, stride=1, padding=2, bias=bias),
            nn.Tanh()
        )
        self.SCA3 =  RCG_SCA(out_channel3*3, ratio=1, bias=True)
        # 多尺度后细化
        self.convBDAM3 = nn.Sequential(
            nn.Conv2d(out_channel3*3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
            nn.Tanh()
        )
        self.detailAM3 = nn.Sequential(
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias)
        )
        # 第三层 encoder


        # # 细化
        # self.SCA_RGB =  RCG_SCA(out_channel3, ratio=1, bias=True)
        # self.convBDAM_RGB = nn.Sequential(
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias), # before detail before multiConv
        #     nn.Tanh()
        # )
        # self.detailAM_RGB = nn.Sequential(
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias)
        # )

        # # ConfidenceGuided
        # self.confidenceGuided = nn.Sequential(
        #     nn.Conv2d(9, out_channel3, kernel_size=7, stride=1, padding=3, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=5, stride=1, padding=2, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3, kernel_size=3, stride=1, padding=1, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3, out_channel3//2, kernel_size=1, stride=1, padding=0, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=7, stride=1, padding=3, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=5, stride=1, padding=2, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, out_channel3//2, kernel_size=3, stride=1, padding=1, bias=bias), 
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3//2, 3, kernel_size=3, stride=1, padding=1, bias=bias), 
        #     nn.Sigmoid()
        # )


        # Decoder 第一层
        # 上采样 + 注意力
        # self.convUp1 = nn.Sequential(
        #     nn.Conv2d(out_channel3, out_channel3*4, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel3*4, out_channel3*4, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.PixelShuffle(upscale_factor=2)  # 通道数 512/4=128 , HxW -> 2Hx2W
        # )
        self.SCA_up1 = RCG_SCA(out_channel3+out_channel2, ratio=1, bias=True)
        # 细化
        self.convUpToRefine11 = nn.Sequential(
            nn.Conv2d(out_channel3+out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        # self.detailUp11 = nn.Sequential(
        #     nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        # )
        self.convUpToRefine12 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        self.detailUp12 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2, out_channel2, kernel_size=3, stride=1, padding=1, bias=bias)
        )


        # Decoder 第二层
        # 上采样 + 注意力
        self.convUp2 = nn.Sequential(
            nn.Conv2d(out_channel2, out_channel2*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel2*4, out_channel2*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(upscale_factor=2)  # 通道数 512/4=128 , HxW -> 2Hx2W
        )
        self.SCA_up2 = RCG_SCA(out_channel2+out_channel1, ratio=1, bias=True)
        # 细化
        self.convUpToRefine21 = nn.Sequential(
            nn.Conv2d(out_channel2+out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        # self.detailUp21 = nn.Sequential(
        #     nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        # )
        self.convUpToRefine22 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        self.detailUp22 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1, out_channel1, kernel_size=3, stride=1, padding=1, bias=bias)
        )


        # Decoder 第三层
        # 上采样 + 注意力
        self.convUp3 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel1*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1*4, out_channel1*4, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelShuffle(upscale_factor=2)  # 通道数 512/4=128 , HxW -> 2Hx2W
        )
        self.SCA_up3 = RCG_SCA(out_channel1+out_channel1//2, ratio=1, bias=True)
        # 细化
        self.convUpToRefine31 = nn.Sequential(
            nn.Conv2d(out_channel1+out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        # self.detailUp31 = nn.Sequential(
        #     nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
        #     nn.Tanh(),
        #     nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        # )
        self.convUpToRefine32 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh()
        )
        self.detailUp32 = nn.Sequential(
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(out_channel1//2, out_channel1//2, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        # 该分支最终输出
        self.outConv = nn.Sequential(
            nn.Conv2d(out_channel1//2, 3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Tanh(),
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, im1, im1_wb, im1_gc, im1_his):
        # 第一层 encoder   # 缺一个concat输入 inputX
        # 多尺度前细化
        ToRefine = self.convBDBM1(torch.cat([im1_wb-im1, im1_gc-im1, im1_his-im1], dim=1))
        Refined = self.detailBM1(ToRefine)
        Refined_res1 = ToRefine + Refined
    # DQ1
        DQ1_A = self.convBDBM1_DQ1_A(torch.cat([im1_wb-im1, im1_gc-im1, im1_his-im1], dim=1))
        DQ1_B = self.convBDBM1_DQ1_B(torch.cat([im1_gc-im1, im1_his-im1, im1_wb-im1], dim=1))
        DQ1_C = self.convBDBM1_DQ1_C(torch.cat([im1_his-im1, im1_wb-im1, im1_gc-im1], dim=1))
        DQ1_SCA = self.SCA_DQ1(torch.cat([DQ1_A, DQ1_B, DQ1_C], dim=1))
        DQ1_all = self.convBDAM1_DQ1_all(DQ1_SCA)
        DQ1_Refined = self.detailBM1_DQ1(DQ1_all)
        DQ1_Refined = DQ1_all + DQ1_Refined
    # DQ1
        # 多尺度卷积
        x1 = self.multiConv11(Refined_res1)
        x3 = self.multiConv31(Refined_res1)
        x5 = self.multiConv51(Refined_res1)
        mt_SCA = self.SCA1(torch.cat([x1, x3, x5], dim=1))
        # 多尺度后细化
        ToRefine = self.convBDAM1(mt_SCA)
        Refined = self.detailAM1(ToRefine)
        Refined = ToRefine + Refined
        # 最大池化
        res1 = self.pool(Refined)
    # DQ2
        DQ1_Refined1 = self.pool(DQ1_Refined)
        DQ1_A_1 = self.convBDBM1_DQ1_A_1(DQ1_Refined1)
        DQ1_B_1 = self.convBDBM1_DQ1_B_1(DQ1_Refined1)
        DQ1_C_1 = self.convBDBM1_DQ1_C_1(DQ1_Refined1)
        DQ1_1_SCA = self.SCA_DQ2(torch.cat([DQ1_A_1, DQ1_B_1, DQ1_C_1], dim=1))
        DQ1_all_1 = self.convBDAM1_DQ1_all_1(DQ1_1_SCA)
        DQ1 = self.detailBM1_DQ1_1(DQ1_all_1)
        DQ1 = DQ1_all_1 + DQ1
    # DQ2
        

        # 第二层 encoder
        scale2 = self.pool(torch.cat([im1_wb-im1, im1_gc-im1, im1_his-im1], dim=1))
        # 多尺度前细化
        ToRefine = self.convBDBM2(scale2)
        Refined = self.detailBM2(ToRefine)
        Refined = ToRefine + Refined
        # 多尺度卷积
        x1 = self.multiConv12(torch.cat([res1, Refined], dim=1))
        x3 = self.multiConv32(torch.cat([res1, Refined], dim=1))
        x5 = self.multiConv52(torch.cat([res1, Refined], dim=1))
        mt_SCA = self.SCA2(torch.cat([x1, x3, x5], dim=1))
        # 多尺度后细化
        ToRefine = self.convBDAM2(mt_SCA)
        Refined = self.detailAM2(ToRefine)
        Refined = ToRefine + Refined
        # 最大池化
        res2 = self.pool(Refined)
    # DQ3
        DQ1_1 = self.pool(DQ1)
        DQ2_A = self.convBDBM2_DQ2_A(DQ1_1)
        DQ2_B = self.convBDBM2_DQ2_B(DQ1_1)
        DQ2_C = self.convBDBM2_DQ2_C(DQ1_1)
        DQ2_SCA = self.SCA_DQ3(torch.cat([DQ2_A, DQ2_B, DQ2_C], dim=1))
        DQ2_all = self.convBDAM2_DQ2_all(DQ2_SCA)
        DQ2 = self.detailBM2_DQ2(DQ2_all)
        DQ2 = DQ2_all + DQ2
    # DQ3


        # 第三层 encoder
        scale3 = self.pool(scale2)
        # 多尺度前细化
        ToRefine = self.convBDBM3(scale3)
        Refined = self.detailBM3(ToRefine)
        Refined = ToRefine + Refined
        # 多尺度卷积
        x1 = self.multiConv13(torch.cat([res2, Refined], dim=1))
        x3 = self.multiConv33(torch.cat([res2, Refined], dim=1))
        x5 = self.multiConv53(torch.cat([res2, Refined], dim=1))
        mt_SCA = self.SCA3(torch.cat([x1, x3, x5], dim=1))
        # 多尺度后细化
        ToRefine = self.convBDAM3(mt_SCA)
        Refined = self.detailAM3(ToRefine)
        # Refined = ToRefine + Refined
        res3 = ToRefine + Refined
        # # 最大池化
        # res3 = self.pool(Refined)


        # # 细化
        # SCA_RGB = self.SCA_RGB(res3)
        # ToRefine = self.convBDAM_RGB(SCA_RGB)
        # Refined = self.convBDAM_RGB(ToRefine)
        # Fusion_Refined = ToRefine + Refined

        # # ConfidenceGuided
        # weight = self.pool(scale3)
        # weight = self.confidenceGuided(weight)
        # out_wb, out_gc, out_histeq = torch.split(weight, 1, dim=1)  # 沿着通道维度拆分为三个单通道
        # res_CG = Fusion_Refined * out_wb + Fusion_Refined * out_gc + Fusion_Refined * out_histeq


        # Decoder 第一层
        # up1 = self.convUp1(res3)
        if res3.size(2) != res2.size(2) or res3.size(3) != res2.size(3):
            res3 = F.interpolate(res3, size=(res2.size(2), res2.size(3)), mode='bilinear', align_corners=True)
        up1_SCA = self.SCA_up1(torch.cat([res3, res2], dim=1))
    # DQ3
        # 细化
        up1ToRefine = self.convUpToRefine11(up1_SCA)
        # 确保 DQ2 和 up1ToRefine 尺寸一致
        if DQ2.size(2) != up1ToRefine.size(2) or DQ2.size(3) != up1ToRefine.size(3):
            DQ2 = F.interpolate(DQ2, size=(up1ToRefine.size(2), up1ToRefine.size(3)), mode='bilinear', align_corners=True)
        DQ2_res = DQ2 * up1ToRefine + up1ToRefine
    # DQ3
        up1ToRefine = self.convUpToRefine12(DQ2_res)
        up1Refined = self.detailUp12(up1ToRefine)
        up1Refined = up1ToRefine + up1Refined


        # Decoder 第二层
        up2 = self.convUp2(up1Refined)
        # 确保 up2 和 res1 尺寸一致
        if up2.size(2) != res1.size(2) or up2.size(3) != res1.size(3):
            up2 = F.interpolate(up2, size=(res1.size(2), res1.size(3)), mode='bilinear', align_corners=True)
        up2_SCA = self.SCA_up2(torch.cat([up2, res1], dim=1))
    # DQ2
        up2ToRefine = self.convUpToRefine21(up2_SCA)
        # 确保 DQ1 和 up1ToRefine 尺寸一致
        if DQ1.size(2) != up2ToRefine.size(2) or DQ1.size(3) != up2ToRefine.size(3):
            DQ1 = F.interpolate(DQ1, size=(up2ToRefine.size(2), up2ToRefine.size(3)), mode='bilinear', align_corners=True)
        DQ1_res = DQ1 * up2ToRefine + up2ToRefine
    # DQ2
        up2ToRefine = self.convUpToRefine22(DQ1_res)
        up2Refined = self.detailUp22(up2ToRefine)
        up2Refined = up2ToRefine + up2Refined

        # Decoder 第三层
        up3 = self.convUp3(up2Refined)
        # 确保 up3 和 Refined_res1 尺寸一致
        if up3.size(2) != Refined_res1.size(2) or up3.size(3) != Refined_res1.size(3):
            up3 = F.interpolate(up3, size=(Refined_res1.size(2), Refined_res1.size(3)), mode='bilinear', align_corners=True)
        up3_SCA = self.SCA_up3(torch.cat([up3, Refined_res1], dim=1))
    # DQ1
        up3ToRefine = self.convUpToRefine31(up3_SCA)
        # 确保 DQ1_ 和 up1ToRefine 尺寸一致
        if DQ1_Refined.size(2) != up3ToRefine.size(2) or DQ1_Refined.size(3) != up3ToRefine.size(3):
            DQ1_Refined = F.interpolate(DQ1_Refined, size=(up3ToRefine.size(2), up3ToRefine.size(3)), mode='bilinear', align_corners=True)
        DQ1_Refined_res = DQ1_Refined * up3ToRefine + up3ToRefine
    # DQ1
        up3ToRefine = self.convUpToRefine32(DQ1_Refined_res)
        up3Refined = self.detailUp32(up3ToRefine)
        up3Refined = up3ToRefine + up3Refined
        
        # 该分支最终输出
        final_output = self.outConv(up3Refined)

        return final_output
