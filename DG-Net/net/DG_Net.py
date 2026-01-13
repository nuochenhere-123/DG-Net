# 论文中使用的RCG-Net
import torch
import torch.nn as nn
import kornia.color as color
# from net.HVI_transform import RGB_HVI
# from net.transformer_utils import *
from net.DG_utils import *

class DGNet(nn.Module):
    def __init__(self, 
                 channels=[32, 64, 128],
                 bias = True        
                 ):
        super(DGNet, self).__init__()
        
        # Reflection Estimation
        self.reflection = encoderRGBLAB(channels, bias=True)
        
        # illumation Estimation
        self.illumation = encoderRGBLAB(channels, bias=True)
        
        # noise Estimation
        self.noise = encoderNoise(channels, bias=True)
        
        self.ReLU = nn.ReLU()
                
    def forward(self, im1, im1_wb, im1_gc, im1_his):
        
        LAB1 = color.rgb_to_lab(im1)
        LAB2 = color.rgb_to_lab(im1_wb)
        LAB3 = color.rgb_to_lab(im1_gc)
        LAB4 = color.rgb_to_lab(im1_his)
        reflection_out = self.reflection(im1, im1_wb, im1_gc, im1_his)
        illumation_out = self.illumation(LAB1, LAB2, LAB3, LAB4)
        noise_out = self.noise(im1, im1_wb, im1_gc, im1_his)
        
        output_rgb_RL = self.ReLU((reflection_out * illumation_out))

        output_rgb = torch.clamp(output_rgb_RL - noise_out, min=0.0, max=1.0)
        output_raw = torch.clamp(output_rgb_RL + noise_out, min=0.0, max=1.0)

        return output_rgb, illumation_out, output_raw
        # return output_rgb, illumation_out, output_raw, reflection_out, noise_out



