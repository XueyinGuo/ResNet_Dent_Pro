from util import data_path, train_path, test_path, train_test_rate
import os
import random
import shutil
from PIL import Image


def resize(image_name):
    image_pil = Image.open(image_name)
    pil_resize = image_pil.resize(size=(224, 224))  # bicubic ,resample=Image.BILINEAR
    return pil_resize


def copy_image(source, des, image_name):
    resized_img = resize(os.path.join(source, image_name))
    resized_img.save(os.path.join(des, image_name))


# 从 data_path 中所有的图片按照配置的比例 生成训练数据集和测试数据集
# # 按照比例生成的数据后，目录树如下
# 分类dataset：
# data_path:
#     |--dent:
#         |--.jpg
#     |--protrusion
#         |--.jpg
#     |--good
#         |--.jpg
# train:
#     |--dent:
#         |--.jpg
#     |--protrusion
#         |--.jpg
#     |--good
#         |--.jpg
# test:
#     |--dent:
#         |--.jpg
#     |--protrusion
#         |--.jpg
#     |--good
#         |--.jpg
def generate():
    if os.path.exists(train_path):
        shutil.rmtree(train_path)

    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    folders = os.listdir(data_path)
    train_rate = train_test_rate[0]
    for folder in folders:
        cur_dir = os.path.join(data_path, folder)
        train_des_dir = os.path.join(train_path, folder)
        test_des_dir = os.path.join(test_path, folder)
        if not os.path.exists(train_des_dir):
            os.makedirs(train_des_dir)
        if not os.path.exists(test_des_dir):
            os.makedirs(test_des_dir)
        images = os.listdir(os.path.join(data_path, folder))
        for image in images:
            rand = random.randint(0, 100)
            if rand > train_rate:
                copy_image(cur_dir, test_des_dir, image)
            else:
                copy_image(cur_dir, train_des_dir, image)


if __name__ == '__main__':
    generate()
