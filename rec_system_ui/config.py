import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别映射 (A-Z)
NUM_CLASSES = 62
IDX_TO_CLASS =  {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm',
    49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
    58: 'w', 59: 'x', 60: 'y', 61: 'z'
}
# 模型名称常量 (用于ComboBox和逻辑判断)
RESNET18 = "ResNet18"
MOBILENETV2 = "MobileNetV2"
CLAN = "CharsLightAttentionNet"

# 模型列表
MODEL_OPTIONS = [RESNET18, MOBILENETV2, CLAN]

# 自定义模型权重路径 (如果存在)
CUSTOM_MODEL_PATH = "../myCnn/v1/best_model.pth"

# 预处理参数 (重要：需要根据实际训练调整！)
# 这些是示例值，请替换为训练时使用的真实值
PREPROCESS_PARAMS = {
    "resnet18": {
        "input_size": 224,
        "input_channels": 3,
        "mean": [0.5, 0.5, 0.5], # 示例值
        "std": [0.5, 0.5, 0.5]   # 示例值
    },
    "mobilenetv2": {
        "input_size": 224,
        "input_channels": 3,
        "mean": [0.5, 0.5, 0.5], # 示例值
        "std": [0.5, 0.5, 0.5]   # 示例值
    },
    "CharsLightAttentionNet": {
        "input_size": 224,
        "input_channels": 3,
        "mean": [0.5, 0.5, 0.5], # 示例值
        "std": [0.5, 0.5, 0.5]   # 示例值
    }
}