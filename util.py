# 按照比例划分测试数据为 [训练数据,测试数据]
# 此时是按照 80%:20%来划分
# 划分数据集的代码在 split_data_train_test.py
train_test_rate = [80, 20]

# 配置图片所在文件夹，
# 此路径的作用是拼接成 data_path 这个变量
# data_path 用来存放已经分好类的训练集图片
# 此时的目录树结构为
# data_path:
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
path = "/home/guoxy/data/"
data_path = path + "分类dataset/"

# 按照比例生成的数据后，目录树如下
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

train_path = path + "train/"
test_path = path + "test/"

# 指定使用哪个权重文件
weight_file = "92.3%_80:20.pth"

# 配判定为 gray 的阈值，阈值越高 gray 区域越多
# 阈值越小 gray 区域越少
gray_threshold = 0.6

# 验证图片的路径
verify_path = "/home/guoxy/data/verify/"

GOOD = 1
DENT = 0
PROTRUSION = 2
GRAY = 3

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 变量名字已经能说明作用了
path_to_save_ng_images = "NG_IMAGES/"
path_to_save_gray_images = "GRAY_IMAGES/"
path_to_save_good_images = "GOOD_IMAGES/"
path_to_log = "log/"
