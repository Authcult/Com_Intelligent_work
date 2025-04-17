import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.se(x)
        return x * weights

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block with SE Attention."""
    def __init__(self, in_channels, out_channels, stride, expansion_factor=4, use_se=True):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = (self.stride == 1) and (in_channels == out_channels)

        hidden_dim = in_channels * expansion_factor
        self.use_se = use_se

        layers = []
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        if self.use_se:
            # 对于通道数很少的情况，可以适当增大 reduction_ratio 或固定一个较小值
            se_reduction = 8 if hidden_dim < 64 else 4
            layers.append(SEBlock(hidden_dim, reduction_ratio=se_reduction))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# --- 优化的 CharsLightAttentionNet ---
class CharsLightAttentionNet(nn.Module):
    def __init__(self, num_classes=62, input_channels=3, input_size=64, width_mult=0.75, use_se=True):
        """
        Optimized for smaller datasets like Chars74k (e.g., 64x64 input).

        Args:
            num_classes (int): Number of character classes. (e.g., 62 for Eng + Digits)
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            input_size (int): Expected input image size (e.g., 64). Affects stem stride.
            width_mult (float): Controls the channel width. Values < 1 make it lighter.
            use_se (bool): Whether to include SE blocks.
        """
        super().__init__()

        # --- Adapt Stem based on input size ---
        # If input size is small (e.g., <= 64), use stride 1 initially to preserve resolution
        initial_stride = 1 if input_size <= 64 else 2
        first_conv_out_channels = int(16 * width_mult) # Reduced initial channels

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, first_conv_out_channels, kernel_size=3,
                      stride=initial_stride, padding=1, bias=False),
            nn.BatchNorm2d(first_conv_out_channels),
            nn.ReLU6(inplace=True)
        )
        current_channels = first_conv_out_channels

        # --- Reduced and Shallower Block Configs ---
        # Config format: [expansion_factor, out_channels, num_blocks, stride]
        # Fewer stages, fewer blocks, fewer channels, less downsampling
        block_configs = [
            # t, c,  n, s
            [4, 24,  2, 2],  # Stage 1 (Downsamples if initial_stride was 1, else maintains)
            [4, 48,  3, 2],  # Stage 2 (Downsamples)
            [4, 64,  2, 1],  # Stage 3 (Maintains resolution)
            # Removed the deeper stages from the original design
        ]

        # --- Build Stages ---
        stages = []
        for t, c, n, s in block_configs:
            # Ensure stride isn't too aggressive if input is already small
            # If initial stride was 1, the first stage's stride 2 is the first downsample.
            # If initial stride was 2, maybe make the first stage stride 1? Or keep it. Let's keep it simple for now.
            actual_stride = s

            output_c = int(c * width_mult)
            # Ensure output channels are at least 1, maybe round to nearest 8 for efficiency? Not strictly needed here.
            output_c = max(1, output_c)

            for i in range(n):
                stride = actual_stride if i == 0 else 1
                stages.append(InvertedResidualBlock(current_channels, output_c, stride,
                                                    expansion_factor=t, use_se=use_se))
                current_channels = output_c
        self.stages = nn.Sequential(*stages)

        # --- Simplified Head ---
        # Reduced channels before GAP
        last_stage_channels = current_channels
        # Reduced expansion before classifier
        head_channels = int(max(128, 256 * width_mult)) # Significantly reduced head channels

        self.head = nn.Sequential(
            # Optional: Small 1x1 conv before GAP
            nn.Conv2d(last_stage_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU6(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.3), # Slightly increased dropout for smaller dataset
            nn.Conv2d(head_channels, num_classes, kernel_size=1) # Use Conv1x1 as FC
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = x.view(x.size(0), -1) # Flatten for classification
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
