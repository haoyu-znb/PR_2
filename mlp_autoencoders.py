import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    全连接层的残差块，用于构建深层 MLP
    结构: Input -> Linear -> BN -> LeakyReLU -> Linear -> BN -> Add Input -> LeakyReLU
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dims=[1024, 512], dropout=0.0):
        """
        完整版 MLP 编码器
        Args:
            input_dim (int): 输入特征维度
            z_dim (int): 潜在空间维度
            hidden_dims (list): 隐藏层维度列表，例如 [512, 256]
            dropout (float): Dropout 比率
        """
        super(MLPEncoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # 1. 构建特征提取层 (逐层压缩)
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        # 2. 可选：加入残差块增强特征表达 (如果网络较深)
        # 这里默认加一层残差块来增强鲁棒性
        layers.append(ResidualBlock(current_dim, dropout))
            
        self.encoder = nn.Sequential(*layers)
        
        # 3. 输出 VAE 的均值和对数方差
        # 这一步是 VAE 的核心，将特征映射为分布参数
        self.fc_mu = nn.Linear(current_dim, z_dim)
        self.fc_logvar = nn.Linear(current_dim, z_dim)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征向量 (Batch, Input_Dim)
        Returns:
            mu: 潜在分布的均值
            logvar: 潜在分布的对数方差
        """
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, output_dim, hidden_dims=[512, 1024], dropout=0.0):
        """
        完整版 MLP 解码器
        Args:
            z_dim (int): 潜在空间维度
            output_dim (int): 输出维度 (必须等于 Encoder 的 input_dim)
            hidden_dims (list): 隐藏层维度列表 (通常与 Encoder 相反)
            dropout (float): Dropout 比率
        """
        super(MLPDecoder, self).__init__()
        
        layers = []
        current_dim = z_dim
        
        # 1. 构建重构层 (逐层放大)
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        # 2. 输出层
        # 还原回原始数据维度
        layers.append(nn.Linear(current_dim, output_dim))
        
        # 注意：这里不加 Sigmoid 或 Tanh，因为特征向量的值域可能是任意实数
        # 如果数据归一化到了 [0,1]，可以在外部加 Sigmoid
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """
        前向传播
        Args:
            z: 采样得到的潜在变量
        Returns:
            recon_x: 重构后的特征向量
        """
        recon_x = self.decoder(z)
        return recon_x