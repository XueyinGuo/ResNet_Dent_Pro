import os
import cv2

from PIL import Image

Image.MAX_IMAGE_PIXELS = 2450000000


def CutDent():
    filepath = "/home/yanni/data/cuntian_all_data/dent/第二次/负样本/NG"
    destpath = '/home/yanni/data/msy_img/dent/resize/'
    print('\t ' + destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    print(filepath, "      ", destpath)
    pathDir = os.listdir(filepath)
    i = 0
    dir_num = 0;
    for allDir in pathDir:

        child = os.path.join(filepath, allDir)
        image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        high = sp[0]  # 图像的高度（行 范围）
        width = sp[1]  # 图像的宽度（列 范围）

        a = 0  # x start
        b = 500  # x end
        c = 0  # y start
        d = 500  # y end
        while d <= high:
            while b <= width:

                cropImg1 = image[c:d, a:b]  # 裁剪图像
                dest = os.path.join(destpath, "dentNGVersion2_" + str(i) + ".jpg")
                cv2.imwrite(dest, cropImg1)
                i = i + 1
                # cv2.imshow("img", cropImg1)
                # cv2.waitKey(0)
                a += 450
                b += 450
                if b == width + 450:
                    break
                if b > width:
                    a = width - 500
                    b = width

                if i % 5001 == 0:
                    dir_num += 1
                    destpath = '/home/yanni/data/msy_img/dent/resize' + str(dir_num) + '/'
                    print('\t '+destpath)
                    if not os.path.exists(destpath):
                        os.makedirs(destpath)
            if d == high:
                break
            c += 450
            d += 450
            a = 0
            b = 500
            if d > high:
                d = high
                c = high - 500
    CutDent_xinzhuijia(i, dir_num)


def CutDent_xinzhuijia(i, dir_num):
    dir_num += 1;
    print("chu li 新追加    id: " + str(i) + "   dir_num: " + str(dir_num))
    filepath = "/home/yanni/data/cuntian_all_data/dent/第二次/负样本/新追加"
    destpath = '/home/yanni/data/msy_img/dent/resize' + str(dir_num) + '/'
    print('\t ' + destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    print(filepath, "      ", destpath)
    pathDir = os.listdir(filepath)
    for allDir in pathDir:

        child = os.path.join(filepath, allDir)
        image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        high = sp[0]  # 图像的高度（行 范围）
        width = sp[1]  # 图像的宽度（列 范围）

        a = 0  # x start
        b = 500  # x end
        c = 0  # y start
        d = 500  # y end
        while d <= high:
            while b <= width:

                cropImg1 = image[c:d, a:b]  # 裁剪图像
                dest = os.path.join(destpath, "dentNGVersion2" + str(i) + ".jpg")
                cv2.imwrite(dest, cropImg1)
                i = i + 1
                # cv2.imshow("img", cropImg1)
                # cv2.waitKey(0)
                a += 450
                b += 450
                if b == width + 450:
                    break
                if b > width:
                    a = width - 500
                    b = width

                if i % 5001 == 0:
                    dir_num += 1
                    destpath = '/home/yanni/data/msy_img/dent/resize' + str(dir_num) + '/'
                    print('\t '+destpath)
                    if not os.path.exists(destpath):
                        os.makedirs(destpath)
            if d == high:
                break
            c += 450
            d += 450
            a = 0
            b = 500
            if d > high:
                d = high
                c = high - 500
    CutDent_zui_xin_shu_ju(i, dir_num)


def CutDent_zui_xin_shu_ju(i, dir_num):
    dir_num += 1;
    print("最新追加   id: " + str(i) + "   dir_num: " + str(dir_num))
    filepath = "/home/yanni/data/cuntian_all_data/dent/第二次/负样本/最新追加/NG"
    destpath = '/home/yanni/data/msy_img/dent/resize' + str(dir_num) + '/'
    print('\t ' + destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    print(filepath, "      ", destpath)
    pathDir = os.listdir(filepath)
    for allDir in pathDir:

        child = os.path.join(filepath, allDir)
        image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        high = sp[0]  # 图像的高度（行 范围）
        width = sp[1]  # 图像的宽度（列 范围）

        a = 0  # x start
        b = 500  # x end
        c = 0  # y start
        d = 500  # y end
        while d <= high:
            while b <= width:

                cropImg1 = image[c:d, a:b]  # 裁剪图像
                dest = os.path.join(destpath, "dentNGVersion2" + str(i) + ".jpg")
                cv2.imwrite(dest, cropImg1)
                i = i + 1
                # cv2.imshow("img", cropImg1)
                # cv2.waitKey(0)
                a += 450
                b += 450
                if b == width + 450:
                    break
                if b > width:
                    a = width - 500
                    b = width

                if i % 5001 == 0:
                    dir_num += 1
                    destpath = '/home/yanni/data/msy_img/dent/resize' + str(dir_num) + '/'
                    print('\t ' + destpath)
                    if not os.path.exists(destpath):
                        os.makedirs(destpath)
            if d == high:
                break
            c += 450
            d += 450
            a = 0
            b = 500
            if d > high:
                d = high
                c = high - 500


def CutProtrusion():
    filepath = "/home/yanni/data/cuntian_all_data/Protrusion/NG/颜色正常"
    destpath = '/home/yanni/data/msy_img/protrusion/resize/'
    print('\t ' + destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    print(filepath, "      ", destpath)
    pathDir = os.listdir(filepath)
    i = 0
    dir_num = 0;
    for allDir in pathDir:

        child = os.path.join(filepath, allDir)
        image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        high = sp[0]  # 图像的高度（行 范围）
        width = sp[1]  # 图像的宽度（列 范围）

        a = 0  # x start
        b = 500  # x end
        c = 0  # y start
        d = 500  # y end
        while d <= high:
            while b <= width:
                cropImg1 = image[c:d, a:b]  # 裁剪图像
                # cv2.imshow("img", cropImg1)
                # cv2.waitKey(0)
                dest = os.path.join(destpath, "protrusionNGVersion2" + str(i) + ".jpg")
                cv2.imwrite(dest, cropImg1)  # 写入图像路径
                i = i + 1
                a += 450
                b += 450
                if b == width + 450:
                    break
                if b > width:
                    a = width - 500
                    b = width

                if i % 5001 == 0:
                    dir_num += 1
                    destpath = '/home/yanni/data/msy_img/protrusion/resize' + str(dir_num) + '/'
                    print('\t ' + destpath)
                    if not os.path.exists(destpath):
                        os.makedirs(destpath)

            if d == high:
                break
            c += 450
            d += 450
            a = 0
            b = 500
            if d > high:
                d = high
                c = high - 500

    CutProtrusionColored(i, dir_num)


def CutProtrusionColored(i, dir_num):
    dir_num += 1;
    print("颜色异常    id: " + str(i) + "   dir_num: " + str(dir_num))
    filepath = "/home/yanni/data/cuntian_all_data/Protrusion/NG/颜色异常"
    destpath = '/home/yanni/data/msy_img/protrusion/resize' + str(dir_num) + '/'
    print('\t ' + destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    print(filepath, "      ", destpath)
    pathDir = os.listdir(filepath)
    for allDir in pathDir:

        child = os.path.join(filepath, allDir)
        image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        high = sp[0]  # 图像的高度（行 范围）
        width = sp[1]  # 图像的宽度（列 范围）

        a = 0  # x start
        b = 500  # x end
        c = 0  # y start
        d = 500  # y end
        while d <= high:
            while b <= width:
                cropImg1 = image[c:d, a:b]  # 裁剪图像
                # cv2.imshow("img", cropImg1)
                # cv2.waitKey(0)
                dest = os.path.join(destpath, "protrusionNGVersion2Colored" + str(i) + ".jpg")
                cv2.imwrite(dest, cropImg1)  # 写入图像路径
                i = i + 1
                a += 450
                b += 450
                if b == width + 450:
                    break
                if b > width:
                    a = width - 500
                    b = width

                if i % 5001 == 0:
                    dir_num += 1
                    destpath = '/home/yanni/data/msy_img/protrusion/resize' + str(dir_num) + '/'
                    print('\t ' + destpath)
                    if not os.path.exists(destpath):
                        os.makedirs(destpath)

            if d == high:
                break
            c += 450
            d += 450
            a = 0
            b = 500
            if d > high:
                d = high
                c = high - 500
    CutProtrusion_zui_xin_zhui_jia(i, dir_num)


def CutProtrusion_zui_xin_zhui_jia(i, dir_num):
    dir_num += 1;
    print("最新追加    id: " + str(i) + "   dir_num: " + str(dir_num))
    filepath = "/home/yanni/data/cuntian_all_data/Protrusion/NG/最新追加/NG"
    destpath = '/home/yanni/data/msy_img/protrusion/resize' + str(dir_num) + '/'
    print('\t ' + destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    print(filepath, "      ", destpath)
    pathDir = os.listdir(filepath)
    for allDir in pathDir:

        child = os.path.join(filepath, allDir)
        image = cv2.imread(child)
        sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
        high = sp[0]  # 图像的高度（行 范围）
        width = sp[1]  # 图像的宽度（列 范围）

        a = 0  # x start
        b = 500  # x end
        c = 0  # y start
        d = 500  # y end
        while d <= high:
            while b <= width:
                cropImg1 = image[c:d, a:b]  # 裁剪图像
                # cv2.imshow("img", cropImg1)
                # cv2.waitKey(0)
                dest = os.path.join(destpath, "protrusionNGVersion2Colored" + str(i) + ".jpg")
                cv2.imwrite(dest, cropImg1)  # 写入图像路径
                i = i + 1
                a += 450
                b += 450
                if b == width + 450:
                    break
                if b > width:
                    a = width - 500
                    b = width

                if i % 5001 == 0:
                    dir_num += 1
                    destpath = '/home/yanni/data/msy_img/protrusion/resize' + str(dir_num) + '/'
                    print('\t ' + destpath)
                    if not os.path.exists(destpath):
                        os.makedirs(destpath)

            if d == high:
                break
            c += 450
            d += 450
            a = 0
            b = 500
            if d > high:
                d = high
                c = high - 500


if __name__ == '__main__':
    CutDent()
    CutDent_zui_xin_shu_ju(160688, 27)
    CutProtrusion()
    CutProtrusion_zui_xin_zhui_jia(153424 ,28)
