import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mlp_dualvae_modules import MLPConsistencyAE

class MembershipCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(MembershipCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        # dims[0] 是输入维度
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return self.get_norm_outputs(h)
    
    def get_norm_outputs(self, inputs):
        norm = torch.norm(inputs, p=3, dim=1, keepdim=True)
        outputs = inputs / norm
        outputs = torch.relu(outputs) 
        return outputs

class net(nn.Module):
    def __init__(self, num_views, num_layer, dims, num_classes):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes  
        self.e = 1e-11
        
        # 解析维度
        self.input_dims = [dims[i][0] for i in range(num_views)]
        
        # --- [核心修改] 自动适配网络规模 ---
        # 逻辑：检测最大输入维度。
        max_dim = max(self.input_dims)
        
        # [修改点] 将阈值从 200 提高到 1000
        # 目的：强制 PIE (维度484, 样本680) 使用小网络，防止过拟合。
        # LandUse (59) 和 Scene (59) 依然会使用小网络。
        if max_dim >= 1000:
            # 只有当维度极其巨大时，才启用大网络
            self.z_dim = 512
            self.hidden_dims = [1024, 512]
            print(f"--- [Auto-Config] Huge Feature Dim ({max_dim} >= 1000). Using Large Net: Hidden=[1024, 512] ---")
        else:
            # PIE (484), LandUse (59), Scene (59) 都会走这里 -> 小网络
            self.z_dim = 128
            self.hidden_dims = [256, 128]
            print(f"--- [Auto-Config] Standard/Small Dim ({max_dim} < 1000). Using Compact Net: Hidden=[256, 128] ---")

        # 初始化 DualVAE
        self.dual_vae = MLPConsistencyAE(
            views=num_views,
            input_dims=self.input_dims,
            z_dim=self.z_dim,
            hidden_dims=self.hidden_dims,  # 使用自动匹配的参数
            kld_weight=0.00025,
            mask_view_ratio=0.2
        )
        
        # 初始化 FUML 分类器
        dims = np.repeat(dims, num_layer, axis=1) 
        self.MembershipCollectors = nn.ModuleList([
            MembershipCollector(dims[i], self.num_classes) for i in range(self.num_views)
        ])

    def forward(self, data_list, label=None, test=False): 
        # --- Stage 1: DualVAE 增强与修复 ---
        vae_loss = 0
        device = self.dual_vae.encoders[0].encoder[0].weight.device

        # 1. 预处理：归一化
        processed_data_list = []
        for i in range(self.num_views):
            x = data_list[i]
            
            # 确保是 Tensor
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
            else:
                x = x.to(device, dtype=torch.float32)
            
            if x.dim() == 1: x = x.unsqueeze(0)
            
            # 实例级 Min-Max 归一化
            min_val = x.min(dim=1, keepdim=True)[0]
            max_val = x.max(dim=1, keepdim=True)[0]
            x_norm = (x - min_val) / (max_val - min_val + 1e-8)
            
            processed_data_list.append(x_norm)

        if not test: 
            # 训练模式：加噪声 (Masking)
            masked_data_list = []
            for x in processed_data_list:
                # Mask 保持 0.2
                mask = torch.rand(x.shape, device=device) > 0.2 
                masked_data_list.append(x * mask.float())
            
            recons, z, mu, logvar = self.dual_vae(masked_data_list)
            
            # 计算 VAE Loss
            recon_loss = 0
            for i in range(self.num_views):
                recon_loss += F.mse_loss(recons[i], processed_data_list[i], reduction='mean')
            
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            vae_loss = recon_loss + self.dual_vae.kld_weight * kld_loss
            
        else: 
            # 测试模式：不加噪声
            recons, z, mu, logvar = self.dual_vae(processed_data_list)
            vae_loss = 0

        # --- 残差连接 (Residual Connection) ---
        # 无论 VAE 训练得如何，原始特征必须保留，重构特征作为补充
        fuml_inputs = []
        for i in range(self.num_views):
            combined_input = processed_data_list[i] + recons[i]
            fuml_inputs.append(combined_input)

        # --- Stage 2: FUML 分类 ---
        Weight, Membership, Credibility, Uncertainty, ConflictDegree = dict(), dict(), dict(), dict(), dict()

        if not test and label is not None:
            one_hot_labels = F.one_hot(label, self.num_classes)

        Weights, MMLogit, MMcrediblity = 0, 0, 0

        for view in range(self.num_views):
            Membership[view] = self.MembershipCollectors[view](fuml_inputs[view])
            Credibility[view] = self.get_test_credibility(Membership[view]) if test else self.get_train_credibility(Membership[view], one_hot_labels)            
            Uncertainty[view] = self.get_fuzzyUncertainty(Credibility[view])

        for view in range(self.num_views):
            conflictDegree = 0
            for v in range(self.num_views):
                if self.num_views > 1:
                    conflictDegree += self.get_ConflictDegree(Membership[view], Membership[v]) * (1/(self.num_views - 1)) 

            ConflictDegree[view] = conflictDegree
            Weight[view] =  (1 - Uncertainty[view]) * (1 - ConflictDegree[view]) + self.e 

        Weights = [Weight[key] for key in sorted(Weight.keys())]
        Weights = torch.stack(Weights)
        Weights = torch.softmax(Weights, dim=0)
        
        for view in range(self.num_views):
            MMLogit += Weights[view] * Membership[view]

        MMcrediblity = self.get_test_credibility(MMLogit) if test else self.get_train_credibility(MMLogit, one_hot_labels)
        MMuncertainty = self.get_fuzzyUncertainty(MMcrediblity)

        return Credibility, MMcrediblity, MMuncertainty, vae_loss 

    # --- 辅助函数 ---
    def get_train_credibility(self, predict, labels):
        top1Possibility = (predict*(1-labels)).max(1)[0].reshape([-1,1])
        labelPossibility = (predict*labels).max(1)[0].reshape([-1,1])
        neccessity = (1-labelPossibility)*(1-labels) + (1-top1Possibility)*labels
        conf = (predict + neccessity)/2
        return conf

    def get_test_credibility(self, membershipDegree): 
        if membershipDegree.shape[1] > 1:
            top2MembershipDegree = torch.topk(membershipDegree, k=2, dim=1, largest=True, sorted=True)[0]
            secMaxMembershipDegree = torch.where(membershipDegree == top2MembershipDegree[:,0].unsqueeze(1), top2MembershipDegree[:,1].unsqueeze(1), top2MembershipDegree[:,0].unsqueeze(1))
            confidence = (membershipDegree + 1 - secMaxMembershipDegree) / 2
        else:
            confidence = membershipDegree
        return confidence

    def get_fuzzyUncertainty(self, credibility):
        nonzero_indices = torch.nonzero(credibility)
        class_num = credibility.shape[1] 
        if len(nonzero_indices) > 1:
            H = torch.sum((-credibility*torch.log(credibility+self.e) - (1-credibility)*torch.log(1-credibility+self.e)), dim=1, keepdim=True)
            H = H / (class_num * torch.log(torch.tensor(2)))
        else:
            H = torch.tensor(0).unsqueeze(0)
        return H

    def get_ConflictDegree(self, vector1, vector2):
        distance = 1 - F.cosine_similarity(vector1, vector2, dim=1, eps=1e-8)
        distance = distance.view(-1, 1) 
        return distance