import time
import os

# 在 predict.py 中调用的一个函数
# 用来创建日志文件和保存问题图片的文件夹
def create_dirs_files(path_to_log, path_to_save_gray_images,
                      path_to_save_ng_images, path_to_save_good_images):
    date = time.strftime("%Y-%m-%d", time.localtime())

    today_ng_log = date + "-NG.txt"
    today_gray_log = date + "-GRAY.txt"
    today_good_log = date + "-GOOD.txt"
    if not os.path.exists(path_to_log):
        os.mkdir(path_to_log)
    txt = os.listdir(path_to_log)
    have_today_ng_log = False
    have_today_gray_log = False
    have_today_good_log = False

    for file in txt:
        if file == today_ng_log:
            have_today_ng_log = True
        if file == today_gray_log:
            have_today_gray_log = True
        if file == today_good_log:
            have_today_good_log = True

    if (have_today_ng_log == False):
        os.mknod(path_to_log + today_ng_log)
    if (have_today_gray_log == False):
        os.mknod(path_to_log + today_gray_log)
    if (have_today_good_log == False):
        os.mknod(path_to_log + today_good_log)

    if not os.path.exists(path_to_save_gray_images):
        os.makedirs(path_to_save_gray_images)

    if not os.path.exists(path_to_save_ng_images):
        os.makedirs(path_to_save_ng_images)

    if not os.path.exists(path_to_save_good_images):
        os.makedirs(path_to_save_good_images)


    f_ng = open(path_to_log + today_ng_log, "a")
    f_gray = open(path_to_log + today_gray_log, "a")
    f_good = open(path_to_log + today_good_log, "a")
    return f_ng, f_gray, f_good
