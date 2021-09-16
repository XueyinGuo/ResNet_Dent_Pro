import torch
import torchvision
from torchvision import datasets, transforms
import os
import time
from PIL import Image
import shutil
from pathlib import Path
import sys
import pretrainedmodels

weight_path = "se_resnext50_32x4d_98.9.pth"
PATH = "/home/msy/AIpic/Cutting dirt/第二次/mixOK"
VALUE_Gray = 0.9
COPY_FILE = False

if COPY_FILE:
    P_path = "/home/msy/AIpic/Cutting dirt/2_result_P"
    N_path = "/home/msy/AIpic/Cutting dirt/2_result_N"
    Gray_path = "/home/msy/AIpic/Cutting dirt/2_result_Gray"

    if Path(P_path).exists() == 1:
        print("The output folder already exists!")
        sys.exit()
    else:
        os.mkdir(P_path)
        os.mkdir(N_path)
        os.mkdir(Gray_path)

# save test log file---------------------
localtime = time.localtime(time.time())
f = ""
for i in range(0, 5):
    f += str(localtime[i]) + "_"
f += "Back_log.txt"
file = open(f, 'w')
# ---------------------------------------

# model=torchvision.models.resnext101_32x8d()
model = pretrainedmodels.se_resnext50_32x4d()
model.last_linear = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
model.load_state_dict(torch.load(weight_path))

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

total = 0
positive = 0
negative = 0
gray = 0

totalStart = start = time.time()

for root, dirs, files in os.walk(PATH):
    for filename in files:

        start = time.time()
        total += 1
        print("check:", total)

        img = Image.open(root + '/' + filename)
        img_ = data_transform(img).unsqueeze(0)  # 拓展维度

        outputs = model(img_.to(device))
        softmax = torch.softmax(outputs, dim=1)
        confidence = torch.max(softmax, dim=1)[0].item()
        predict = torch.max(softmax, dim=1)[1].item()

        end = time.time()
        timeStr = ""
        for i in range(0, 6):
            timeStr += str(localtime[i]) + "."

        if confidence < VALUE_Gray:
            gray += 1
            file.write(PATH + filename + " | gray | " + str(round(confidence, 4)) + " | tact:" + str(
                round(end - start, 3)) + "| time:" + timeStr + "\n")
            if COPY_FILE:
                shutil.copy(os.path.join(root, filename), os.path.join(Gray_path, filename))
        else:
            if predict == 0:
                negative += 1
                file.write(PATH + filename + " | negative | " + str(round(confidence, 4)) + " | tact:" + str(
                    round(end - start, 3)) + "| time:" + timeStr + "\n")
                if COPY_FILE:
                    shutil.copy(os.path.join(root, filename), os.path.join(N_path, filename))
            else:
                positive += 1
                file.write(PATH + filename + " | positive | " + str(round(confidence, 4)) + " | tact:" + str(
                    round(end - start, 3)) + "| time:" + timeStr + "\n")
                if COPY_FILE:
                    shutil.copy(os.path.join(root, filename), os.path.join(P_path, filename))

totalEnd = time.time()
totalTime = round(totalEnd - totalStart, 3)
print("total:", total, " positive:", positive, " gray:", gray, " negative:", negative)
file.close()
file = open(f, 'r+')
content = file.read()
file.seek(0, 0)
file.write("total:" + str(total) + "  positive:" + str(positive) + "   gray:" + str(gray) + "  negative:" + str(
    negative) + "  time:" + str(totalTime) + "\n\n" + content)
file.close()
