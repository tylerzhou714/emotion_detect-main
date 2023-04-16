from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QImage
from PyQt5.QtCore import QCoreApplication, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore
import sys
import cv2

from video import EmotionDetector


from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QSplitter
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QImage
from PyQt5.QtCore import QCoreApplication, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore
import sys
import cv2

from video import EmotionDetector


class DisplayLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)
        self.emotion_detector = EmotionDetector()

    def display_image(self, img):
        img = self.emotion_detector.detect_emotion(img)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def display_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def display_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.emotion_detector.detect_emotion(frame)

        height, width, channel = frame.shape
        bytes_per_line = channel * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if hasattr(self, 'cap'):
            self.cap.release()
        event.accept()


class Ico(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        font_z = QFont()
        font_z.setFamily('华文行楷')
        font_z.setBold(True)
        font_z.setPointSize(9)
        font_z.setWeight(50)

        self.setGeometry(300, 300, 1024, 768)
        self.setWindowTitle('课堂微表情识别系统')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./utils/bg.jpg")))
        self.setPalette(palette)

        x1 = QPushButton('打开相机', self)
        x1.setFont(font_z)
        x1.setIcon(QIcon("./utils/camera.jpg"))
        x1.clicked.connect(self.open_camera)

        x2 = QPushButton('打开图片', self)
        x2.setFont(font_z)
        x2.setIcon(QIcon("./utils/img.jpg"))
        x2.clicked.connect(self.open_image)

        x3 = QPushButton('打开视频', self)
        x3.setFont(font_z)
        x3.setIcon(QIcon("./utils/video.jpg"))
        x3.clicked.connect(self.open_video)

        qbtn = QPushButton('退出', self)
        qbtn.setFont(font_z)
        qbtn.setIcon(QIcon("./utils/exit.jpg"))
        qbtn.clicked.connect(QCoreApplication.instance().quit)

        control_layout = QVBoxLayout()
        control_layout.addWidget(x1)
        control_layout.addWidget(x2)
        control_layout.addWidget(x3)
        control_layout.addWidget(qbtn)
        control_layout.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        self.display_label = DisplayLabel()

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(self.display_label)
        main_splitter.setSizes([self.width() // 4, self.width() * 3 // 4])

        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)

        self.setLayout(main_layout)

        self.show()

    def open_camera(self):
        self.display_label.display_camera()

    def open_image(self):
        if hasattr(self.display_label, 'timer'):
            self.display_label.timer.stop()
            self.display_label.cap.release()

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            img = cv2.imread(file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_label.display_image(img)

    def open_video(self):
        if hasattr(self.display_label, 'timer'):
            self.display_label.timer.stop()
            self.display_label.cap.release()

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi);;All Files (*)",
                                                   options=options)
        if file_name:
            self.display_label.display_video(file_name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ico()
    sys.exit(app.exec_())
