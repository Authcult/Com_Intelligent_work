import torch.nn as nn
import torch


class SEBlock(nn.Module) :
    def __init__(self, channels, reduction=4) :
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x) :
        weights = self.se(x)
        return x * weights


class InvertedResidualSE(nn.Module) :
    def __init__(self, inp, oup, stride, expand_ratio) :
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1 :
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            SEBlock(hidden_dim),  # 插入SE模块
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x) :
        if self.use_res :
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_SE(nn.Module) :
    def __init__(self, num_classes=1000) :
        super().__init__()
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        # Inverted Residual Blocks配置
        self.inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # 构建带SE的块
        input_channel = 32
        for t, c, n, s in self.inverted_residual_setting :
            output_channel = c
            for i in range(n) :
                stride = s if i == 0 else 1
                self.features.append(InvertedResidualSE(input_channel, output_channel, stride, t))
                input_channel = output_channel
        # 末尾层（修复后）
        self.features.append(nn.Conv2d(input_channel, 1280, 1, bias=False))
        self.features.append(nn.BatchNorm2d(1280))
        self.features.append(nn.ReLU6())

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x) :
        x = self.features(x)
        x = self.classifier(x)
        return x