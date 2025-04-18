import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow # 从主窗口模块导入 MainWindow 类

if __name__ == '__main__':
    # 创建 PyQt 应用实例
    app = QApplication(sys.argv)

    # 创建主窗口实例
    main_win = MainWindow()

    # 显示主窗口
    main_win.show()

    # 进入 Qt 应用的事件循环
    sys.exit(app.exec_())