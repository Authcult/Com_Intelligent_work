import os
import numpy as np
from PIL import Image, ImageOps
import io
from albumentations.pytorch import ToTensorV2
import albumentations as A

from PyQt5.QtCore import QBuffer

from myCnn.baseline.resnet18 import resnet18
from myCnn.baseline.mobilenet_v2 import mobilenet_v2
from myCnn.baseline.LeNet5 import LeNet5
from myCnn.v1.CBAMNet_Lite import CharsLightAttentionNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from config import (
    DEVICE, NUM_CLASSES, RESNET18, MOBILENETV2, CLAN,
    CUSTOM_MODEL_PATH, PREPROCESS_PARAMS, IDX_TO_CLASS
)


# --- 模型加载 ---
def load_selected_model(model_name):
    """根据名称加载模型结构，并可能加载预训练权重"""
    print(f"尝试加载模型: {model_name} 到设备: {DEVICE}")
    model = None
    params_key = ""
    try:
        if model_name == RESNET18:
            params_key = "resnet18"
            model = resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
            print("加载 ResNet18 结构。需要您提供针对字母训练的权重。")
            # 加载权重:
            model_path = "../myCnn/baseline_weight/resnet18_best_model.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print(f"已加载 ResNet18 权重: {model_path}")


        elif model_name == MOBILENETV2:
            params_key = "mobilenetv2"
            model = mobilenet_v2(pretrained=True)
            print("加载 MobileNetV2 结构。需要您提供针对字母训练的权重。")
            model_path = "../myCnn/baseline_weight/mobilenetv2_best_model.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print(f"已加载 MobileNetV2 权重: {model_path}")


        elif model_name == CLAN:
            params_key = "CharsLightAttentionNet"
            model = CharsLightAttentionNet(num_classes=NUM_CLASSES)
            # 尝试加载自定义模型的权重
            if os.path.exists(CUSTOM_MODEL_PATH):
                model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE))
                print(f"成功加载自定义模型权重: {CUSTOM_MODEL_PATH}")

        else:
             print(f"错误：未知的模型名称 '{model_name}'")
             return None, None # 返回 None 表示失败

        model.to(DEVICE)
        model.eval() # 设置为评估模式
        print(f"{model_name} 模型加载完成。")
        return model, PREPROCESS_PARAMS[params_key]

    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None, None # 返回 None 表示失败


# --- 图像预处理 ---
def preprocess_image(img_pil):
    class AlbumentationsTransform :
        def __init__(self) :
            self.transform = A.Compose([
                A.Resize(32, 32),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

        def __call__(self, img) :
            img = np.array(img)
            # img = np.array(img.convert('L'))
            return self.transform(image=img)['image']

    transform = AlbumentationsTransform()

    img_tensor = transform(img_pil)

    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor



# --- 识别逻辑 ---
def recognize_letter(model, image_qimage, params):
    """使用加载的模型识别 QImage 中的字母"""
    if model is None or params is None:
        return "错误：模型或参数未加载", []

    try:
        # 1. 将 QImage 转换为 PIL Image
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        # 保存为 PNG 格式到内存缓冲区
        image_qimage.save(buffer, "PNG")
        buffer.seek(0)

        # 读取字节数据并转换为 PIL Image
        pil_image = Image.open(io.BytesIO(buffer.data())).convert('RGB') # 先转RGB，预处理会处理灰度

        # 检查图像是否几乎为空白 (全白或接近全白)
        # 转为灰度图检查，稍微提高阈值容忍一些噪点

        # 2. 预处理图像
        input_tensor = preprocess_image(pil_image)
        input_tensor = input_tensor.to(DEVICE)

        # 3. 模型推理
        with torch.no_grad():
            outputs = model(input_tensor)
            top_p, top_class_idx = torch.topk(outputs, k=3, dim=1)
            print("前3个预测结果:", top_class_idx, top_p)
            print(IDX_TO_CLASS)

        # 4. 处理结果
        top_p = top_p.squeeze().cpu().numpy()
        top_class_idx = top_class_idx.squeeze().cpu().numpy()

        results = []
        for i in range(len(top_p)):
            pred_idx = top_class_idx[i]
            pred_class = IDX_TO_CLASS.get(pred_idx, "未知") # 从配置中获取类别名

            results.append({"class": pred_class})

        return "识别成功", results

    except Exception as e:
        print(f"识别过程中发生错误: {e}")
        return f"识别错误: {e}", []
