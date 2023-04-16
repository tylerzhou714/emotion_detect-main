import cv2
import dlib
import numpy as np
import torch
from facenet_pytorch.models.mtcnn import MTCNN

from repvgg import create_RepVGG_A0
from torchvision import transforms
import win32ui

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(device=device)


class EmotionDetector:
    def __init__(self, confidence=0.85):
        self.deploy_model = create_RepVGG_A0(deploy=False).to(device)
        self.detector = dlib.get_frontal_face_detector()
        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
        self.deploy_model.load_state_dict(torch.load('./model/best_model.pth', map_location=device))
        self.deploy_model.eval()
        self.emo_labels = {0: 'understand', 1: 'doubt', 2: 'careless'}
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.confidence = confidence

    def detect_faces(self, image):
        """
        使用dlib的人脸检测器检测图片中的人脸
        :param image: 输入图片
        :return: 返回人脸的矩形框坐标
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        return rects

    def detect_emotion(self, image):
        rects = self.detect_faces(image)
        for rect in rects:
            startX, startY, endX, endY = rect.left(), rect.top(), rect.right(), rect.bottom()

            face = image[startY: endY, startX: endX]

            face = self.trans(face)
            face = torch.unsqueeze(face, dim=0).to(device)
            self.deploy_model.eval()
            out = self.deploy_model(face)
            prediction = torch.argmax(out, 1)

            emo = self.emo_labels[int(prediction)]
            if out[0][prediction] < self.confidence:
                emo = "Unknown"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, '%s' % emo, (startX + 30, startY + 30), self.font, 1, (144, 238, 144), 2)
        return image

    def detect_emotion_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_emotion(frame)

            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = EmotionDetector()
    detector.detect_emotion_from_video("./video/190314102306987969.mp4")  # 替换为实际的视频路径
