import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp_autoencoders import MLPEncoder, MLPDecoder
from .moe import Moe

# ==========================================
# 工具函数区域 (直接内嵌，防止找不到 utils 报错)
# ==========================================

def mask_view(views, mask_ratio, view_num):
    """
    模拟视图缺失 (View-Mask)
    随机将某些视图的数据全部置零
    """
    masked_views = []
    for v in views:
        # 生成一个 (B, 1) 的 mask
        # 这里的逻辑是：生成随机数，大于 mask_ratio 的保留，小于的置零
        mask = torch.rand(v.shape[0], 1).to(v.device) > mask_ratio
        masked_views.append(v * mask.float())
    return masked_views

def mask_vector(x, mask_ratio):
    """
    模拟特征噪声 (Vector-Mask / Patch-Mask)
    随机将特征向量中的某些维度置零
    """
    if mask_ratio <= 0:
        return x
    
    B, D = x.shape
    # 生成随机掩码矩阵
    mask = torch.rand(B, D).to(x.device) > mask_ratio
    return x * mask.float()

# ==========================================
# 核心模型区域
# ==========================================

class MLPConsistencyAE(nn.Module):
    def __init__(self, views, input_dims, z_dim, hidden_dims=[1024, 512], kld_weight=0.00025, mask_view_ratio=0.5):
        """
        基于 MLP 和 MoE 的一致性 VAE (完整版)
        """
        super().__init__()
        self.views = views
        self.input_dims = input_dims
        self.z_dim = z_dim
        self.kld_weight = kld_weight
        self.mask_view_ratio = mask_view_ratio
        
        # 1. 多视图编码器列表
        self.encoders = nn.ModuleList([
            MLPEncoder(input_dim=dim, z_dim=z_dim, hidden_dims=hidden_dims) 
            for dim in input_dims
        ])
        
        # 2. 专家模型 (MoE)
        # 注意：使用 hidden_dims[-1] 作为输入，2*z_dim 作为输出
        self.moe = Moe(
            views=views, 
            input_dim=hidden_dims[-1], 
            output_dim=2 * z_dim 
        )
        
        # 3. 多视图解码器列表
        self.decoders = nn.ModuleList([
            MLPDecoder(z_dim=z_dim, output_dim=dim, hidden_dims=list(reversed(hidden_dims))) 
            for dim in input_dims
        ])

    def forward(self, Xs):
        """
        前向传播：编码 -> MoE融合 -> 采样 -> 解码
        """
        # 1. 编码
        latents = []
        for i, x in enumerate(Xs):
            # 调用 MLPEncoder 的 .encoder 部分提取特征
            feat = self.encoders[i].encoder(x) 
            latents.append(feat)
            
        # 2. MoE 融合
        moe_out = self.moe(latents)
        
        # 3. 分割均值和方差
        mu, logvar = torch.split(moe_out, self.z_dim, dim=1)
        
        # 4. 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 5. 解码
        recons = []
        for i, decoder in enumerate(self.decoders):
            recons.append(decoder(z))
            
        return recons, z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def get_loss(self, Xs, epoch=0):
        """
        保留这个方法以备不时之需 (虽然 model.py 可能自己实现了 mask)
        """
        # 1. View-Mask
        Xs_masked_view = mask_view(Xs, self.mask_view_ratio, self.views)
        
        # 2. Vector-Mask (固定 0.2 或随 epoch 变化)
        Xs_masked = [mask_vector(x, mask_ratio=0.2) for x in Xs_masked_view]
        
        # 3. 前向传播
        recons, z, mu, logvar = self.forward(Xs_masked)
        
        # 4. 计算 Loss
        # KL Loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
        # Recon Loss
        recon_loss = 0
        for i, (recon, x_original) in enumerate(zip(recons, Xs)):
            recon_loss += F.mse_loss(recon, x_original, reduction='mean')
            
        total_loss = recon_loss + self.kld_weight * kld_loss
        
        return total_loss, {
            "cons_recon_loss": recon_loss.item(),
            "cons_kld_loss": kld_loss.item()
        }


class MLPIVAE(nn.Module):
    def __init__(self, views, input_dims, z_dim, hidden_dims=[1024, 512], kld_weight=0.00025):
        """
        独立 VAE (Independent VAE) - 完整保留
        """
        super().__init__()
        self.views = views
        self.encoders = nn.ModuleList([
            MLPEncoder(input_dim=dim, z_dim=z_dim, hidden_dims=hidden_dims) 
            for dim in input_dims
        ])
        self.decoders = nn.ModuleList([
            MLPDecoder(z_dim=z_dim, output_dim=dim, hidden_dims=list(reversed(hidden_dims))) 
            for dim in input_dims
        ])
        self.kld_weight = kld_weight

    def get_loss(self, Xs):
        loss = 0
        for i in range(self.views):
            # 这里的 MLPEncoder 需要返回 mu, logvar
            # 注意：我们的 MLPEncoder.forward 直接返回 mu, logvar，所以可以直接调用
            mu, logvar = self.encoders[i](Xs[i])
            
            std = torch.exp(0.5 * logvar)
            z = torch.randn_like(std) * std + mu
            
            recon = self.decoders[i](z)
            
            recon_loss = F.mse_loss(recon, Xs[i], reduction='mean')
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            
            loss += recon_loss + self.kld_weight * kld_loss
            
        return loss