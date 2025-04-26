# svm+hog图像分类
import numpy as np
from albumentations import ToTensorV2
import albumentations as A
from sklearn import svm
from sklearn.metrics import accuracy_score ,f1_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from myCnn.utils import split_dataset

BATCH_SIZE=128


# 加载数据集
class AlbumentationsTransform:
    def __init__(self):
        self.transform=A.Compose([
            A.Resize(28, 28),
            A.Rotate(limit=15, p=0.5),
            A.Affine(translate_percent=(0.1,0.1),p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.5,),std=(0.5,)),
            ToTensorV2()
        ])
    def __call__(self, img):
        # img=np.array(img)
        img = np.array(img.convert('L'))
        return self.transform(image=img)['image']

# 提取HOG特征
def extract_hog_features(images):
    hog_features = []
    for img in images:
        fd, _ = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True, channel_axis=None)
        hog_features.append(fd)
    return np.array(hog_features)

# 数据预处理
def preprocess_data(dataloader):
    images, labels = [], []
    for img, label in dataloader:
        img = img.numpy().squeeze()  # 去掉多余的维度
        label = label.numpy()
        images.append(img)
        labels.append(label)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

# 主函数
def main():
    # 加载数据
    transform = AlbumentationsTransform()

    train_loader, val_loader, test_loader, full_dataset = split_dataset(
        root_dir="../../emnist_png_balanced",
        # root_dir="EnglishImg/EnglishImg/English/Img/GoodImg/Bmp",
        transform=transform,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=BATCH_SIZE,
        shuffle=True,
        random_seed=42
    )

    # 打印数据集大小
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 预处理数据
    X_train_raw, y_train = preprocess_data(train_loader)
    X_test_raw, y_test = preprocess_data(test_loader)

    # 提取HOG特征
    X_train = extract_hog_features(X_train_raw)
    X_test = extract_hog_features(X_test_raw)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练SVM模型
    svm_clf = svm.SVC(kernel='linear', C=1.0, probability=True)
    svm_clf.fit(X_train, y_train)

    # 预测和评估
    y_train_pred = svm_clf.predict(X_train)
    y_test_pred = svm_clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"训练集准确率: {train_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"训练集 F1 分数: {train_f1:.4f}")
    print(f"测试集 F1 分数: {test_f1:.4f}")

if __name__ == "__main__":
    main()
