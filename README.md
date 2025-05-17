# 基于卷积神经网络的手写英文字母识别系统研究

## 项目简介

本项目实现了一个基于卷积神经网络的手写英文字母识别系统。系统采用 PyQt5 构建用户界面，支持通过手写绘制或上传图片的方式输入字母，并使用我们自己训练好的包含注意力的卷积神经网络模型进行识别。

## 数据集

本项目使用 EMNIST (Extended MNIST) 数据集的 balanced 子集进行训练和测试。该数据集包含各种手写英文字母和数字，为模型提供了丰富的训练样本。

1. [The EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## 系统功能

- **字母输入**：支持两种输入方式
  - 绘制输入：用户可在画布上直接绘制字母
  - 图片上传：支持从本地上传图片文件进行识别
- **模型选择**：提供多种预训练模型可供选择
  - ResNet18
  - MobileNetV2
  - 我们的带注意力机制的 CNN 网络
- **结果展示**：显示模型识别的 Top-3 结果

## 环境需求

可以使用以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

```
Com_Intelligent_work/
├── baseline.ipynb          # 基线模型
├── myCnn.ipynb             # 我们的带注意力机制的 CNN 网络
├── result_plot.ipynb       # 结果可视化
├── requirements.txt        # 项目依赖
├── myCnn/                  # 算法库
│   ├── baseline/           # 基线模型网络结构及训练代码
│   ├── baseline_weight/    # 基线模型权重
│   ├── cnnWithAttention/   # 我们的带注意力机制的 CNN 网络结构及训练代码
│   └── save/               # 我们的带注意力机制的 CNN 网络权重
└── rec_system_ui/          # 字母识别系统
    ├── main.py             # 主程序入口
    ├── main_window.py      # 主窗口实现
    ├── canvas.py           # 画布控件实现
    ├── model_handler.py    # 模型加载和处理
    └── config.py           # 配置文件
```


