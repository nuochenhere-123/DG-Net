import torch
import torch.nn as nn
from einops import rearrange
# from net.transformer_utils import *

# SCA 通道空间注意力，输出与input_x相同size的特征
class RCG_SCA(nn.Module):
    def __init__(self, in_channels, ratio=8, bias=True): # in_channels 输入特征图的通道数；ratio 控制通道注意力中降维的比例
        super(RCG_SCA, self).__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 对每个通道做全局平均池化和最大池化，结果尺寸为 [B, C, 1, 1]
        self.channelAttention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // ratio, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=bias), # kernel_size=1
            nn.Sigmoid()
        )
        
        # Spatial Attention 可考虑换更大感受野
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=bias),  
            nn.GELU(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=bias), # kernel_size=1
            nn.Sigmoid()
        )

    def forward(self, x):
        # ----- Channel Attention -----
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        channel_att = self.channelAttention(torch.cat([avg_out, max_out], dim=1))  # element-wise sum
        x = x * channel_att  # broadcasting

        # ----- Spatial Attention ----- 可考虑在通道注意力前进行空间注意力权重提取
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))  # [B, 1, H, W]
        x = x * spatial_att  # broadcasting

        return x


# Cross Attention Block
class RCG_QKV(nn.Module):
    def __init__(self, dim, ratio=8, bias=True):
        super(RCG_QKV, self).__init__()
        num_heads = dim // 16
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # 自注意力
        original = x

        q = self.q_dwconv(self.q(x))
        # 交叉注意力
        # kv = self.kv_dwconv(self.kv(y))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        # 自注意力
        return out+original
        # return out