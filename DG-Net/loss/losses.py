import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *
import kornia

_reduction_modes = ['none', 'mean', 'sum']

class MSELoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

# 灰度世界
class ColorBalanceLoss(nn.Module):
    """Color Balance Loss.

    Encourages the RGB channels to have similar average intensities.
    Typically used to promote color constancy in image enhancement tasks.

    Args:
        loss_weight (float): Loss weight for color balance loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply across batch.
                         Choices: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ColorBalanceLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: none | mean | sum')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target=None, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (N, 3, H, W). Predicted RGB image.
            target (ignored): Included for compatibility.
            weight (ignored): Not used in this loss.
        """
        # 1. 灰度世界：三个通道均值差最小
        mean_rgb = pred.mean(dim=(2, 3))  # (N, 3)
        gray_mean_loss = torch.mean((mean_rgb - mean_rgb.mean(dim=1, keepdim=True)) ** 2, dim=1)  
        # (N,) 计算每张图像的 RGB 平均值的平均值，并求出三通道分别与均值之差，再求平均

        # Reduction
        if self.reduction == 'mean':
            gray_mean_loss = gray_mean_loss.mean()
        elif self.reduction == 'sum':
            gray_mean_loss = gray_mean_loss.sum()
        # else: 'none' - do nothing

        return self.loss_weight * gray_mean_loss

# 多样性色彩
class ColorVarianceLoss(nn.Module):
    """
    Encourages each RGB channel to have higher variance (color richness/spread).
    Promotes vivid and saturated images with broader color distributions.
    
    Args:
        loss_weight (float): Weight applied to the loss.
        reduction (str): 'mean' | 'sum' | 'none'. Default is 'mean'.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ColorVarianceLoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target=None, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (N, 3, H, W). Predicted RGB image.
        Returns:
            Loss that penalizes low per-channel variance.
        """
        # 2. 方差最大化：通道内方差越大越好
        var_r = pred[:, 0, :, :].var(dim=(1, 2), unbiased=False)  # (N,)
        var_g = pred[:, 1, :, :].var(dim=(1, 2), unbiased=False)
        var_b = pred[:, 2, :, :].var(dim=(1, 2), unbiased=False)

        diversity_loss = (var_r + var_g + var_b) / 3.0

        # # Reduction 求平均后按batch惩罚
        # if self.reduction == 'mean':
        #     diversity_loss = diversity_loss.mean()
        #     var_r = var_r.mean()
        #     var_g = var_g.mean()
        #     var_b = var_b.mean()
        # elif self.reduction == 'sum':
        #     diversity_loss = diversity_loss.sum()
        #     var_r = var_r.sum()
        #     var_g = var_g.sum()
        #     var_b = var_b.sum()

        # 对每张图像独立惩罚/按batch惩罚
        penalty_max_r = torch.clamp(var_r - 6000, min=0) ** 2
        penalty_min_r = torch.clamp(1000 - var_r, min=0) ** 2
        penalty_max_g = torch.clamp(var_g - 5500, min=0) ** 2
        penalty_min_g = torch.clamp(1000 - var_g, min=0) ** 2
        penalty_max_b = torch.clamp(var_b - 5900, min=0) ** 2
        penalty_min_b = torch.clamp(1000 - var_b, min=0) ** 2
        # 惩罚
        penalty_max = (penalty_max_r + penalty_max_g + penalty_max_b) / 3.0
        penalty_min = (penalty_min_r + penalty_min_g + penalty_min_b) / 3.0

        # # 求平均后按batch惩罚：每个batch一个最终 loss
        # total_loss = diversity_loss - penalty_max - penalty_min

        # 对每张图像独立惩罚后求平均
        # 每张图像一个最终 loss
        loss_per_image = diversity_loss - penalty_max - penalty_min
        # Reduction 
        if self.reduction == 'mean':
            total_loss = loss_per_image.mean()
        elif self.reduction == 'sum':
            total_loss = loss_per_image.sum()

        # penalty_max = max(0, diversity_loss-5000)**2  #1
        # penalty_min = min(0, diversity_loss-1500)**2  #1
        # penalty_max = max(0, diversity_loss-4300)
        # penalty_min = min(0, diversity_loss-1200)

        # # Reduction
        # if self.reduction == 'mean':                  #1
        #     diversity_loss = diversity_loss.mean()    #1
        # elif self.reduction == 'sum':                 #1
        #     diversity_loss = diversity_loss.sum()     #1

        # return self.loss_weight * (diversity_loss - penalty_max - penalty_min) #1
        # return self.loss_weight * (diversity_loss + penalty_min - penalty_max)
        return self.loss_weight * total_loss


class HistogramEMDLoss(nn.Module):
    """Histogram Earth Mover's Distance Loss.

    Encourages the histogram of predicted images to match that of reference (e.g. enhanced) images.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'.
        bins (int): Number of histogram bins. Default: 64.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', bins=64):
        super(HistogramEMDLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction: {reduction}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.bins = bins
        # self.emd_loss_fn = kornia.losses.EMD() # 使用 Kornia 提供的可微分 Earth Mover’s Distance（Wasserstein距离）

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): (N, 3, H, W), predicted image [0,1]
            target (Tensor): (N, 3, H, W), histogram-enhanced reference
        """
        N, C, H, W = pred.shape
        device = pred.device
        total_loss = 0.0

        # Per channel histogram and EMD
        for c in range(C):
            # 对每个通道分别计算预测图像和目标图像的直方图
            pred_hist = self._soft_histogram(pred[:, c:c+1], self.bins).to(device)  # (N, bins)
            targ_hist = self._soft_histogram(target[:, c:c+1], self.bins).to(device)
            # 用 Earth Mover’s Distance (EMD) 比较它们，一种衡量两个概率分布之间距离的方法
            total_loss += self.emd_loss_1d(pred_hist, targ_hist, reduction='none') 

        # 对 3 个通道的 EMD 取平均
        loss = total_loss / C

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss

    # torch.histc() 是 非可导 的，不能用于反向传播，因此使用了 高斯平滑核 做 soft-assignment，得到可导版本的 histogram
    def _soft_histogram(self, x, bins, min_val=0.0, max_val=1.0, sigma=0.01):
        """Compute differentiable histogram using Gaussian kernel smoothing."""
        N, _, H, W = x.shape
        x = x.view(N, -1)  # 将图像通道从 [N, 1, H, W] 展平成 [N, H*W]
        # 对每个像素，计算它与每个 bin 的差距，并通过高斯函数 exp(-(x - c)^2 / (2σ^2)) 转为“概率投票”
        # 创建 histogram 的 bin 中心点（等间隔的值）
        centers = torch.linspace(min_val, max_val, steps=bins, device=x.device).view(1, -1)  # [1, B]
        x_expanded = x.unsqueeze(-1)  # [N, HW, 1]
        centers = centers.unsqueeze(1)  # [1, 1, B]
        # 每个像素对所有 bins 的“投票概率”
        weights = torch.exp(-0.5 * ((x_expanded - centers) / sigma)**2)  # [N, HW, B]
        # 对所有像素“投票”结果进行求和
        weights = weights.sum(dim=1)  # [N, B]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6) # 确保每个样本的 histogram 总和为 1，即转为概率直方图，方便计算 KL / EMD 等分布类损失
        # 每张图的通道生成了一条长度为 bins 的 soft histogram
        return weights  # [N, B]
    
    def emd_loss_1d(self, p, q, reduction='mean'):
        """
        计算一维 EMD（Wasserstein 距离），p 和 q 为 soft histogram (N, B)

        Args:
            p: (N, B), predicted histograms
            q: (N, B), target histograms
            reduction: 'mean' | 'sum' | 'none'

        Returns:
            Tensor: EMD distance for each sample
        """
        # 1. 累积分布函数 (CDF)
        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)

        # 2. Wasserstein-1 距离（L1 范数）
        emd = torch.abs(cdf_p - cdf_q).sum(dim=1)  # shape: (N,)

        if reduction == 'mean':
            return emd.mean()
        elif reduction == 'sum':
            return emd.sum()
        else:  # 'none'
            return emd


class LightnessLoss(nn.Module):
    """Lightness Loss.

    Encourages the overall brightness of the image to match a target level.
    Useful for avoiding under- or over-exposed outputs in image enhancement tasks.

    Args:
        target_value (float): The desired mean lightness (0~1). Default: 0.6.
        loss_weight (float): Weight of this loss term. Default: 1.0.
        reduction (str): Reduction method: 'mean', 'sum', or 'none'. Default: 'mean'.
    """
    def __init__(self, target_value=0.6, loss_weight=1.0, reduction='mean'):
        super(LightnessLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction: {reduction}. Use "mean", "sum", or "none".')

        self.target_value = target_value
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target=None, weight=None, **kwargs):
        # Convert to grayscale using luminance formula (ITU-R BT.601)
        gray = 0.299 * pred[:, 0, :, :] + 0.587 * pred[:, 1, :, :] + 0.114 * pred[:, 2, :, :]  # (N, H, W)

        # Build a target tensor of the same shape
        target_gray = torch.full_like(gray, self.target_value)

        # MSE between grayscale image and target lightness
        loss = (gray - target_gray) ** 2  # (N, H, W)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else: 'none' → keep per-pixel loss

        return self.loss_weight * loss


        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight