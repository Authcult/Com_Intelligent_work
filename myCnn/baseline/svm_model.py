import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def train_svm(X_train, y_train, X_val, y_val):
    """
    训练SVM模型

    参数:
    - X_train: 训练集特征
    - y_train: 训练集标签
    - X_val: 验证集特征
    - y_val: 验证集标签

    返回:
    - svm_clf: 训练好的SVM分类器
    - train_acc: 训练集准确率
    - val_acc: 验证集准确率
    """
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 初始化SVM分类器
    svm_clf = svm.SVC(kernel='linear', C=1.0, probability=True)

    # 训练SVM模型
    svm_clf.fit(X_train, y_train)

    # 预测和评估
    y_train_pred = svm_clf.predict(X_train)
    y_val_pred = svm_clf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    val_acc = accuracy_score(y_val, y_val_pred) * 100

    print(f"训练集准确率: {train_acc:.2f}%")
    print(f"验证集准确率: {val_acc:.2f}%")

    return svm_clf, train_acc, val_acc

def flatten_images(dataloader):
    """
    将图像数据展平

    参数:
    - dataloader: DataLoader对象

    返回:
    - images: 展平后的图像特征
    - labels: 对应的标签
    """
    images = []
    labels = []
    for img, label in dataloader:
        # 展平图像
        img = img.view(img.size(0), -1).numpy()  # img.size(0) 是批次大小
        images.append(img)
        # 将标签转换为numpy数组并展平
        labels.append(label.numpy())
    # 合并批次
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels
