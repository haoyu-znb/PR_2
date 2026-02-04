import torch
import torch.nn as nn
import torch.nn.functional as F

class Moe(nn.Module):
    def __init__(self, views, input_dim, output_dim):
        """
        混合专家模型 (Mixture-of-Experts)
        适配浩宇提供的版本
        """
        super(Moe, self).__init__()
        self.num_experts = views
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 隐藏层维度计算逻辑
        self.hidden_dim = (input_dim + output_dim) * 2 // 3

        # 专家网络列表
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            ) for _ in range(self.num_experts)
        ])
        
        # 门控网络
        self.gates = nn.Linear(input_dim * views, self.num_experts)

    def forward(self, Xs):
        """
        Args:
            Xs: list of tensors, 每个 tensor 形状为 (B, D)
        """
        # 1. 拼接所有视图特征用于计算门控权重
        gate_input = torch.cat(Xs, dim=-1)
        
        # 2. 计算权重 (B, views)
        gate_score = F.softmax(self.gates(gate_input), dim=-1) 
        
        # 3. 专家计算
        expers_output = [self.experts[i](Xs[i]) for i in range(self.num_experts)]
        expers_output = torch.stack(expers_output, dim=1)  # (B, views, output_dim)
        
        # 4. 加权融合
        # bmm: (B, 1, views) x (B, views, output_dim) -> (B, 1, output_dim)
        output = torch.bmm(gate_score.unsqueeze(1), expers_output).squeeze(1)

        return output