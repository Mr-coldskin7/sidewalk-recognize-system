import os
import sys
import cv2
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from models.experimental import attempt_load
from utils.general import check_img_size, clip_boxes, non_max_suppression,scale_boxes
from utils.plots import Annotator
import pyttsx3
import time
import line

def speak_label(label):
    # Convert the label text to speech and play it using pyttsx3
    engine = pyttsx3.init()  # Initialize the pyttsx3 engine
    try:
        engine.say(label)  # Have the engine say the label text
        engine.runAndWait()  # Wait for the engine to finish speaking
    except Exception as e:
        print("Error generating speech:", e)

class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.cap = cv2.VideoCapture()
        self.out = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.update_video)
        cudnn.benchmark = True
        weights = 'weights/trafficlight.pt'  # 模型加载路径
        imgsz = 640  # 预测图尺寸大小
        self.conf_thres = 0.25  # NMS置信度
        self.iou_thres = 0.45  # IOU阈值

        # 载入模型
        self.model = attempt_load(weights, device=None, inplace=True, fuse=True)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()  # to FP16

        # 从模型中获取各类别名称
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # 给每一个类别初始化颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.save_path='pyqt_yolo/'
        self.last_speak_time = 0
        self.detected_labels = set()

    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 1200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.video = QtWidgets.QLabel(self.centralwidget)
        self.video.setGeometry(QtCore.QRect(0, 0, 900, 675))
        self.video.setObjectName("video")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.start_video()

    def start_video(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)  # 0代表默认摄像头，如果有多个摄像头可以选择不同的索引
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)  # 设置摄像头宽度
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  # 设置摄像头高度
                self.cap.set(cv2.CAP_PROP_FPS,30)  # 设置摄像头帧率
            self.timer_video.start(30)  # 每隔30毫秒更新一帧视频
        except Exception as e:
            print(f'摄像头打开失败: {e}')

    def stop_video(self):
        self.timer_video.stop()
        self.cap.release()

    def update_video(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # 进行图像处理、目标检测等操作
                origin = np.copy(frame)
                frame = line.show_lane(origin)
                output = np.concatenate((origin, frame), axis=1)
                frame = cv2.resize(frame, (self.imgsz, self.imgsz))
                img = torch.from_numpy(frame).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img = torch.div(img, float(255.0))  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # 修改输入的形状，使其与权重的形状一致
                # 从: [batch_size, height, width, channels]
                # 到: [batch_size, channels, height, width]
                img = img.permute(0, 3, 1, 2)
                pred = self.model(img, augment=False)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # 创建Annotator对象
                annotator = Annotator(frame, line_width=3)
                detected_labels_this_frame = set()
                for i, det in enumerate(pred):  # detections per image
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{self.names[int(cls)]} {conf}'
                            detected_labels_this_frame.add(self.names[int(cls)])
                            # 使用box_label方法绘制检测框和标签
                            annotator.box_label(xyxy, label=label, color=self.colors[int(cls)])
                # 在video标签上显示图像
                self.detected_labels.update(detected_labels_this_frame)
                self.display_image(frame, self.video)
                current_time = time.time()
                if current_time - self.last_speak_time >= 5:
                    # 播报所有不同的标签名称
                    for label in self.detected_labels:
                        speak_label(label)
                        # 更新上次播报的时间
                    self.detected_labels.clear()
                    self.last_speak_time = current_time
            else:
                print('图像读取失败')
        except Exception as e:
            print(f'视频更新失败: {e}')
            self.video.setText(f'WRONG:{e}')

    def display_image(self, frame, label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = pixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.video.setText(_translate("MainWindow", "TextLabel"))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()

    def closeEvent(event):
        ui.stop_video()
        event.accept()

    ui.MainWindow.closeEvent = closeEvent
    ui.setupUi(ui.MainWindow)
    ui.MainWindow.show()

    sys.exit(app.exec_())