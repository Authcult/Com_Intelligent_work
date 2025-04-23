import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import DataLoader


class SimpleAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Squeeze操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作：两个全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 输入形状 [B, C, H, W]
        # Squeeze
        y = self.avg_pool(x).view(b, c)  # [B, C]
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        # Reweight
        return x * y.expand_as(x)  # 广播乘法 [B, C, H, W]

# 自注意力模块
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放系数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        proj_q = self.query(x).view(B, -1, H * W)          # B x C1 x N
        proj_k = self.key(x).view(B, -1, H * W)            # B x C1 x N
        proj_v = self.value(x).view(B, -1, H * W)          # B x C  x N

        attention = torch.bmm(proj_q.permute(0, 2, 1), proj_k)  # B x N x N
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(proj_v, attention.permute(0, 2, 1))     # B x C x N
        out = out.view(B, C, H, W)

        return self.gamma * self.sigmoid(out) + x

# 添加残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        #调用注意力机制
        self.attn = SimpleAttention(out_channels) if use_attention else nn.Identity()
        # self.attn = CBAM(out_channels) if use_attention else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)
        out = self.attn(out)
        return self.relu(out + res)

# 加入倒残差模块（实际没用上）
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=6, stride=1, use_attention=False):
        super().__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 调用注意力模块
        # self.attn = CBAM(out_channels) if use_attention else nn.Identity()
        self.attn = SimpleAttention(out_channels) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.attn(out)
        if self.use_res_connect:
            return x + out
        else:
            return out


class CNNWithAttention(nn.Module) :
    def __init__(self, num_classes, use_attention=True) :
        super().__init__()
        self.use_attention = use_attention  # 添加这行保存参数

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 插入残差和倒残差模块
        if use_attention :
            self.residual = ResidualBlock(64, 64, use_attention=True)
            self.inverted_residual = InvertedResidualBlock(64, 64, expansion_ratio=6, use_attention=True)
            self.attention = SelfAttention2D(64)
        else :
            self.residual = ResidualBlock(64, 64, use_attention=False)
            self.inverted_residual = InvertedResidualBlock(64, 64, expansion_ratio=6, use_attention=False)
            self.attention = nn.Identity()

        self.fcon1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出形状 [B, 64, 1, 1]
            nn.Flatten(),  # 展平为 [B, 64]
            nn.Dropout(0.4),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.fcon2 = nn.Linear(128, num_classes)

    def forward(self, x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        # x = self.MaxPool(x)
        # x = self.inverted_residual(x)
        if self.use_attention :  # 这里现在可以正确访问了
            x = self.attention(x)
        x = self.fcon1(x)
        x = self.fcon2(x)
        return x