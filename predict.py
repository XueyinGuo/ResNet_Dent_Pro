import torch
from torchvision import transforms
import os
import time
from PIL import Image
import pretrainedmodels
from util import weight_file, gray_threshold, verify_path
from util import GOOD, DENT, PROTRUSION, GRAY, COLORS
from util import path_to_save_gray_images, path_to_save_ng_images, \
    path_to_log, path_to_save_good_images
from create_dirs_files import create_dirs_files
import cv2
import datetime


def label_image(pt1, pt2, bottom_left, cv2_image, predict_type):
    label = ""
    if predict_type == DENT:
        label = "dent"
    elif predict_type == PROTRUSION:
        label = "protrusion"
    else:
        label = "gray"
    # 画框 + 写字
    cv2.rectangle(cv2_image, pt1, pt2, COLORS[1], 2)
    cv2.putText(cv2_image, label, bottom_left,
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


def forward_calculate(model, transform, roi):
    img_ = transform(roi).unsqueeze(0)  # 拓展维度
    outputs = model(img_.to(device))
    softmax = torch.softmax(outputs, dim=1)
    # 预测置信度，就是大约有 n% 的概率是 predict 类型
    # predict 有三个可以取值， DENT， PROTRUSION， GOOD
    confidence = torch.max(softmax, dim=1)[0].item()
    predict = torch.max(softmax, dim=1)[1].item()
    '''
    0 = 'dent'
    1 = 'good'
    2 = 'protrusion'
    '''
    # 如果 置信度小于 gray 阈值，那么此滑窗区域即为 gray
    if confidence < gray_threshold:
        return GRAY, confidence
    else:
        # 否则预测结果置信度就比较高，可以直接认定为就是某一类，到底是哪一类具体看 predict 的具体值
        if predict == DENT:
            return DENT, confidence
        elif predict == PROTRUSION:
            return PROTRUSION, confidence
        else:
            return GOOD, confidence


def predict(transform, model, f_ng, f_gray, f_good):
    # 每种图片的数量，其中 total 为图片总数量
    total = 0
    positive = 0
    negative = 0
    gray = 0

    for root, dirs, files in os.walk(verify_path): # 枚举每种图片
    # 此时的目录树结构为
    # verify_path:
    #     |--dent:
    #         |--.jpg
    #         |--.jpg
    #         |--...
    #     |--protrusion
    #         |--.jpg
    #         |--.jpg
    #         |--...
    #     |--good
    #         |--.jpg

        for dir in dirs: # 此时枚举每个文件夹（假设此时应该遍历第一个文件夹--dent）

            cur_path = os.path.join(root, dir)
            images = os.listdir(cur_path)
            for image_name in images: # 枚举 dent 文件夹下的每张图片
                total += 1
                image_path = os.path.join(cur_path, image_name)
                plt_image = Image.open(image_path)  # 读取出图片
                cv2_image = cv2.imread(image_path)
                # sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
                high = plt_image.height  # 图像的高度（行 范围）
                width = plt_image.width  # 图像的宽度（列 范围）

                # 开始滑窗穷举大图
                a = 0  # x start
                b = 500  # x end
                c = 0  # y start
                d = 500  # y end
                log_to = "good"
                cur_log = ""
                start = time.time()
                while d <= high:
                    cur_log += "\t\t\t  |  X0Y0 : ( " + str(a) + ", " + str(c) + " )"
                    while b <= width:
                        box = (a, c, b, d)  # box为4元祖, a c分别为裁剪图片左上角, b d右下角的像素坐标
                        roi = plt_image.crop(box)  # 裁剪图像
                        resized_roi = roi.resize(size=(224, 224))

                        # 计算当前小图的预测结果
                        predict_type, confidence = forward_calculate(model, transform, resized_roi)

                        # 上一个句代码执行完毕之后，现在就有了此滑窗区域是哪一类的判断
                        # 下边的代码就是要在对应的检测结果上画出方框，并且标注缺陷类型
                        # 在 此滑窗中做标记， a c分别为裁剪图片左上角, b d右下角的像素坐标
                        # bottom_left 是滑窗的左下角点，在这个点上写上对应的文字
                        pt1 = (a, c)
                        pt2 = (b, d)
                        bottom_left = (a, d)
                        if predict_type == DENT or predict_type == PROTRUSION:
                            log_to = "ng"
                            label_image(pt1, pt2, bottom_left, cv2_image, predict_type)
                        if predict_type == GRAY:
                            if log_to == "good":
                                log_to = "gray"
                            label_image(pt1, pt2, bottom_left, cv2_image, predict_type)
                        cur_log += "  " + str(format(confidence, '.4f'))  # 添加当前滑窗的日志结果

                        a += 450
                        b += 450
                        if b == width + 450:
                            break
                        if b > width:
                            a = width - 500
                            b = width

                    if d == high:
                        break
                    c += 450
                    d += 450
                    a = 0
                    b = 500
                    if d > high:
                        d = high
                        c = high - 500
                # 计算大图的预测完毕的时间
                cost = time.time() - start

                cur_log += "  \ttcat : " + str(
                    str(format(cost, '.2f'))) + " s   |  time : " + datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')

                # 记录日志
                if log_to == "good":
                    positive += 1
                    f_good.writelines(image_name + "   |   G" + cur_log + '\n')
                    f_good.flush()
                    cv2.imwrite(path_to_save_good_images + image_name, cv2_image)
                if log_to == "ng":
                    negative += 1
                    f_ng.writelines(image_name + "   |   NG" + cur_log + '\n')
                    f_ng.flush()
                    cv2.imwrite(path_to_save_ng_images + image_name, cv2_image)
                if log_to == "gray":
                    gray += 1
                    f_gray.writelines(image_name + "   |   GRAY" + cur_log + '\n')
                    f_gray.flush()
                    cv2.imwrite(path_to_save_gray_images + image_name, cv2_image)
    return total, positive, gray, negative


if __name__ == '__main__':
    # 创建存放日志的文件夹及三个txt日志文件
    # 同时创建三个 文件夹，分别用来存放 被判为 Gray、NG、Good 的三种图片
    f_ng, f_gray, f_good = create_dirs_files(path_to_log, path_to_save_gray_images,
                                             path_to_save_ng_images, path_to_save_good_images)
    # 这是从 util.py 中直接导入的权重文件的名称
    weight_path = weight_file
    # 加载模型
    model = pretrainedmodels.se_resnext50_32x4d()
    model.last_linear = torch.nn.Linear(in_features=2048, out_features=3, bias=True)
    model.load_state_dict(torch.load(weight_path))
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # 开始进行检测
    total, positive, gray, negative = predict(data_transform, model, f_ng, f_gray, f_good)
    print("total : " + str(total))
    print("positive : " + str(positive))
    print("gray : " + str(gray))
    print("negative : " + str(negative))
