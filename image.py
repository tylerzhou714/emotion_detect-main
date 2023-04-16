import cv2
import torch
import numpy as np
from PIL import Image
from repvgg import create_RepVGG_A0
from torchvision import transforms
import win32ui
from facenet_pytorch import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deploy_model = create_RepVGG_A0(deploy=False).to(device)
mtcnn = MTCNN(device=device)

emo_labels = {0: 'understand', 1: 'doubt', 2: 'careless'}

deploy_model.load_state_dict(torch.load('./model/best_model.pth', map_location=device))
deploy_model.eval()
trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
font = cv2.FONT_HERSHEY_SIMPLEX


def openimage():
    dlg = win32ui.CreateFileDialog(1)
    dlg.SetOFNInitialDir('G:\456\4')
    dlg.DoModal()

    imgname = dlg.GetPathName()
    if imgname:
        img = cv2.imread(imgname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(img_pil)

        if boxes is not None:
            for box in boxes:
                startX, startY, endX, endY = box.astype(int)

                image = img[startY:endY, startX:endX]
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image = trans(image)
                image = torch.unsqueeze(image, dim=0).to(device)
                deploy_model.eval()
                out = deploy_model(image)
                prediction = torch.argmax(out, 1)

                emo = emo_labels[int(prediction)]
                print('Emotion : ', emo)

                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(img, '%s' % emo, (startX + 30, startY + 30), font, 1, (144, 238, 144), 2)
        else:
            print("未检测到人脸")

        cv2.imshow("image", img)
        cv2.waitKey(0)
    else:
        print("请输入图片")


if __name__ == '__main__':
    openimage()
