import torch.nn as nn
# 定义CNN模型
class LeNet5(nn.Module):
    def __init__(self, label_num=62):  # 62 类别（0-9, A-Z, a-z）
        super(LeNet5, self).__init__()
        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),  # 1@32×32 → 6@28×28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 6@28×28 → 6@14×14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),  # 6@14×14 → 16@10×10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 16@10×10 → 16@5×5
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 16×5×5 → 120
            nn.ReLU(),
            nn.Linear(120, 84),  # 120 → 84
            nn.ReLU(),
            nn.Linear(84, label_num)  # 84 → label_num
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x
