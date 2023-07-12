import os

import cv2


def resizeImage(file, type, name):
    image = cv2.imread(file, cv2.IMREAD_COLOR)

    # 如果type(image) == 'NoneType',会报错,导致程序中断,所以这里先跳过这些图片,
    # 并记录下来,结束程序后手动修改(删除)

    if image is None:
        pass
    else:
        resizeImg = cv2.resize(image, (1024, 1024))  # 这里改为自己想要的分辨率
        cv2.imwrite("image/data" + str(type) + "/" + name, resizeImg)
        cv2.waitKey(100)

def resizeImage2(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)

    # 如果type(image) == 'NoneType',会报错,导致程序中断,所以这里先跳过这些图片,
    # 并记录下来,结束程序后手动修改(删除)

    if image is None:
        pass
    else:
        resizeImg = cv2.resize(image, (1024, 1024))  # 这里改为自己想要的分辨率
        cv2.imwrite(file, resizeImg)
        cv2.waitKey(100)

root = "image/"

# for i in range(2):
#     _root = root + str(i)
#     file_list = os.listdir(_root)
#     for url in file_list:
#         resizeImage(_root + "/" + url, i, url)
_root = root + "test/raw2"
file_list = os.listdir(_root)
for url in file_list:
    resizeImage2(_root + "/" + url)