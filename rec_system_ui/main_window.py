import sys
import os
from PIL import Image # 需要 Pillow 来处理上传的图片预览

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QTextEdit, QTabWidget, QSizePolicy, QSpacerItem, QStyle
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt, QSize

# 从其他模块导入
from canvas import Canvas
from model_handler import load_selected_model, recognize_letter
from config import DEVICE, MODEL_OPTIONS, IDX_TO_CLASS

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.current_params = None # 存储当前模型的预处理参数
        self.uploaded_image_path = None
        self.drawn_image = None # 存储来自画布的 QImage

        self.init_ui()
        self.load_style() # 加载样式表
        self.load_initial_model() # 初始化时加载第一个模型

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("手写字母识别系统")
        self.setGeometry(150, 150, 600, 700) # 调整窗口大小和位置

        # --- 中心控件和主布局 ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15) # 外边距
        main_layout.setSpacing(10) # 控件间距

        # --- 模型选择 ---
        model_layout = QHBoxLayout()
        model_label = QLabel("选择模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_OPTIONS)
        self.model_combo.currentIndexChanged.connect(self.on_model_change) # 连接信号
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1) # 让下拉框占据更多空间
        main_layout.addLayout(model_layout)

        # --- 输入区域 (Tab) ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- 绘制标签页 ---
        draw_widget = QWidget()
        draw_layout = QVBoxLayout(draw_widget)
        draw_layout.setContentsMargins(5, 5, 5, 5)
        self.canvas = Canvas() # 使用自定义的 Canvas
        draw_layout.addWidget(self.canvas) # Canvas 会自动扩展
        self.tab_widget.addTab(draw_widget, "绘制字母")

        # --- 上传标签页 ---
        upload_widget = QWidget()
        upload_layout = QVBoxLayout(upload_widget)
        upload_layout.setContentsMargins(5, 5, 5, 5)
        upload_layout.setSpacing(8)

        self.upload_button = QPushButton("选择图片文件")
        # 使用Qt内置图标
        upload_icon = self.style().standardIcon(getattr(QStyle, 'SP_DialogOpenButton', QStyle.SP_DirOpenIcon))
        self.upload_button.setIcon(QIcon(upload_icon))
        self.upload_button.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_button, 0, Qt.AlignLeft) # 按钮靠左

        self.image_preview_label = QLabel("尚未上传图片")
        self.image_preview_label.setObjectName("ImagePreviewLabel") # 设置 objectName 以应用样式
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumSize(280, 280) # 设置最小尺寸
        self.image_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # 允许扩展
        upload_layout.addWidget(self.image_preview_label, 1) # 占据剩余空间

        self.tab_widget.addTab(upload_widget, "上传图片")

        # --- 控制按钮区域 ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.clear_button = QPushButton("清除输入")
        clear_icon = self.style().standardIcon(getattr(QStyle, 'SP_DialogResetButton', QStyle.SP_TrashIcon))
        self.clear_button.setIcon(QIcon(clear_icon))
        self.clear_button.clicked.connect(self.clear_input)

        self.recognize_button = QPushButton("开始识别")
        recognize_icon = self.style().standardIcon(getattr(QStyle, 'SP_DialogApplyButton', QStyle.SP_ArrowRight))
        self.recognize_button.setIcon(QIcon(recognize_icon))
        self.recognize_button.clicked.connect(self.recognize)

        # 添加弹性空间将按钮推到右侧
        button_layout.addStretch(1)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.recognize_button)
        main_layout.addLayout(button_layout)

        # --- 结果显示区域 ---
        results_label = QLabel("识别结果 (Top 3):")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(110) # 固定高度
        main_layout.addWidget(results_label)
        main_layout.addWidget(self.results_text)

        # --- 状态栏 ---
        self.statusBar().showMessage("就绪。请选择模型并提供输入。")

    def load_style(self):
        """加载 QSS 样式表"""
        try:
            with open("styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
            print("样式表 'styles.qss' 加载成功。")
        except FileNotFoundError:
            print("警告：未找到样式表 'styles.qss'，将使用默认样式。")
        except Exception as e:
            print(f"加载样式表时出错: {e}")

    def load_initial_model(self):
        """程序启动时加载默认模型"""
        self.on_model_change(0) # 加载下拉框中的第一个模型

    def on_model_change(self, index):
        """处理模型选择变化"""
        model_name = self.model_combo.itemText(index)
        self.statusBar().showMessage(f"正在加载模型: {model_name} ...")
        QApplication.processEvents() # 强制UI更新

        self.current_model, self.current_params = load_selected_model(model_name)

        if self.current_model and self.current_params:
            self.statusBar().showMessage(f"模型 '{model_name}' 加载成功 (设备: {DEVICE})。")
            self.recognize_button.setEnabled(True) # 模型加载成功后启用识别按钮
            self.results_text.setPlaceholderText("请绘制或上传字母，然后点击“开始识别”。")
        else:
            self.statusBar().showMessage(f"加载模型 '{model_name}' 失败。请查看控制台输出。")
            self.results_text.setPlaceholderText(f"加载模型 '{model_name}' 失败，无法进行识别。")
            self.recognize_button.setEnabled(False) # 模型加载失败则禁用识别按钮

    def upload_image(self):
        """处理图片上传"""
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # 根据需要取消注释
        file_path, _ = QFileDialog.getOpenFileName(self, "选择字母图片", "",
                                                  "图片文件 (*.png *.jpeg *.jpg *.bmp);;所有文件 (*)", options=options)
        if file_path:
            self.uploaded_image_path = file_path
            try:
                # 显示预览图
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    raise ValueError("无法加载图片文件")

                # 缩放图片以适应标签，保持纵横比
                scaled_pixmap = pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_preview_label.setPixmap(scaled_pixmap)
                self.image_preview_label.setAlignment(Qt.AlignCenter)
                self.results_text.clear() # 清除旧结果
                self.statusBar().showMessage(f"图片 '{os.path.basename(file_path)}' 已加载。")
                # 切换到上传标签页
                self.tab_widget.setCurrentIndex(1)
                # 清除画布内容，避免混淆
                self.canvas.clear_canvas()
                self.drawn_image = None

            except Exception as e:
                 self.statusBar().showMessage(f"加载预览图片失败: {e}")
                 self.image_preview_label.setText(f"无法预览图片:\n{os.path.basename(file_path)}\n{e}")
                 self.uploaded_image_path = None # 加载失败则重置路径


    def clear_input(self):
        """清除当前输入（画布或上传的图片）和结果"""
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == 0: # 绘制标签页
            self.canvas.clear_canvas()
            self.drawn_image = None
            self.statusBar().showMessage("画布已清除。")
        elif current_tab_index == 1: # 上传标签页
            self.uploaded_image_path = None
            self.image_preview_label.setText("尚未上传图片")
            self.image_preview_label.setPixmap(QPixmap()) # 清除预览图
            self.statusBar().showMessage("上传的图片已重置。")

        self.results_text.clear() # 总是清除结果区域

    def recognize(self):
        """执行识别操作"""
        if self.current_model is None or self.current_params is None:
            self.results_text.setText("错误：模型未成功加载，无法识别。")
            self.statusBar().showMessage("识别失败：模型未加载。")
            return

        image_to_process_qimage = None
        source_description = ""

        current_tab_index = self.tab_widget.currentIndex()

        if current_tab_index == 0: # 绘制标签页
            self.drawn_image = self.canvas.get_image() # 获取当前画布图像
            if self.drawn_image:
                image_to_process_qimage = self.drawn_image
                source_description = "绘制的字母"
            else:
                 self.results_text.setText("请先在画布上绘制一个字母。")
                 self.statusBar().showMessage("输入无效：画布为空。")
                 return

        elif current_tab_index == 1: # 上传标签页
            if self.uploaded_image_path:
                try:
                    # 需要从文件路径加载 QImage 以便传递给识别函数
                    temp_qimage = QImage(self.uploaded_image_path)
                    if temp_qimage.isNull():
                        raise ValueError("无法从文件加载 QImage")
                    image_to_process_qimage = temp_qimage
                    source_description = f"上传的图片 ({os.path.basename(self.uploaded_image_path)})"
                except Exception as e:
                     self.results_text.setText(f"无法加载上传的图片进行识别: {e}")
                     self.statusBar().showMessage("识别失败：无法加载图片。")
                     return
            else:
                self.results_text.setText("请先上传一张图片。")
                self.statusBar().showMessage("输入无效：未上传图片。")
                return

        if image_to_process_qimage is None:
             self.results_text.setText("没有找到有效的输入图像。")
             self.statusBar().showMessage("识别失败：无有效输入。")
             return

        # --- 开始处理和识别 ---
        self.statusBar().showMessage(f"正在识别 {source_description} ...")
        QApplication.processEvents() # 更新UI

        status_msg, results = recognize_letter(self.current_model, image_to_process_qimage, self.current_params)

        # --- 显示结果 ---
        if results:
            result_str = ""
            for i, res in enumerate(results):
                result_str += f"{i+1}. 预测: {res['class']}\n"
            self.results_text.setText(result_str.strip())
            self.statusBar().showMessage("识别完成。")
        else:
            # 显示错误或提示信息
            self.results_text.setText(status_msg)
            if "错误" in status_msg or "失败" in status_msg :
                 self.statusBar().showMessage("识别失败。")
            else: # 比如画布为空的提示
                 self.statusBar().showMessage(status_msg)