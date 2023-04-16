import os
import shutil
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from repvgg import create_RepVGG_A0
from facenet_pytorch import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model = create_RepVGG_A0(deploy=False).to(device)

# 加载训练好的模型
train_model.load_state_dict(torch.load('../model/best_model.pth', map_location=device))
train_model.eval()

mtcnn = MTCNN(device=device)
def predict_image(cropped_face):
    if cropped_face is not None:
        image = Image.fromarray((cropped_face * 255).astype('uint8'))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        output = train_model(image_tensor)
        pred_label = torch.argmax(output, 1)
        label = pred_label.item()
    else:
        label = 2  # 未检测到人脸时将其归类为“漫游”

    return label

emotion_dict = {
    0: "0",
    1: "1",
    2: "2"
}

source_folder = r"G:\emotion_datasets\3"  # 更改为您的未分类图片文件夹路径
destination_folder = r"G:\emotion_datasets"  # 更改为您要将分类后的图片保存的文件夹路径
train_folder = os.path.join(destination_folder, "train")
val_folder = os.path.join(destination_folder, "val")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
for label in emotion_dict.values():
    os.makedirs(os.path.join(train_folder, label), exist_ok=True)
    os.makedirs(os.path.join(val_folder, label), exist_ok=True)

image_names = [image_name for image_name in os.listdir(source_folder) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
random.shuffle(image_names)
split_index = int(len(image_names) * 0.8)

for index, image_name in enumerate(image_names):
    image_path = os.path.join(source_folder, image_name)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    output = train_model(image_tensor)
    pred_label = torch.argmax(output, 1)
    label = pred_label.item()

    if index < split_index:
        shutil.copy(image_path, os.path.join(train_folder, emotion_dict[label], image_name))
    else:
        shutil.copy(image_path, os.path.join(val_folder, emotion_dict[label], image_name))

