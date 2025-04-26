import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别映射 (A-Z)
NUM_CLASSES = 37
IDX_TO_CLASS =  {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'a', 27: 'b', 28: 'd',
    29: 'e', 30: 'f', 31: 'g', 32: 'h',
    33: 'n', 34: 'q', 35: 'r', 36: 't'
}
# 模型名称常量 (用于ComboBox和逻辑判断)
RESNET18 = "ResNet18"
MOBILENETV2 = "MobileNetV2"
CLAN = "CNNWithAttention"

# 模型列表
MODEL_OPTIONS = [RESNET18, MOBILENETV2, CLAN]

# 自定义模型权重路径 (如果存在)
CUSTOM_MODEL_PATH = "../myCnn/cnnWithAttention/cnn_res_attention_best.pth"

# 预处理参数 (重要：需要根据实际训练调整！)
# 这些是示例值，请替换为训练时使用的真实值
PREPROCESS_PARAMS = {
    "resnet18": {
        "model_name": RESNET18,
        "input_size": 224,
        "input_channels": 3,
        "mean": [0.5, 0.5, 0.5], # 示例值
        "std": [0.5, 0.5, 0.5]   # 示例值
    },
    "mobilenetv2": {
        "model_name": MOBILENETV2,
        "input_size": 224,
        "input_channels": 3,
        "mean": [0.5, 0.5, 0.5], # 示例值
        "std": [0.5, 0.5, 0.5]   # 示例值
    },
    "CNNWithAttention": {
        "model_name": CLAN,
        "input_size": 224,
        "input_channels": 3,
        "mean": [0.5, 0.5, 0.5], # 示例值
        "std": [0.5, 0.5, 0.5]   # 示例值
    }
}