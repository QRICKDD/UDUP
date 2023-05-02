import os
import cv2
from AllConfig.GConfig import abspath
dir_path = os.path.join(abspath,'AllData/train')
dir_path2 = os.path.join(abspath,'AllData/train_clip') # 修改为实际路径


# 遍历目录中所有PNG图像
for filename in os.listdir(dir_path):
    if filename.endswith(".png"):
        # 加载图像
        img = cv2.imread(os.path.join(dir_path, filename))
        height, width, _ = img.shape

        # 如果宽或高超过600像素，则切割图像
        if width > 600 or height > 600:
            x_start = 0
            y_start = 0
            count = 1
            while True:
                x_end = min(x_start + 600, width)
                y_end = min(y_start + 600, height)
                if x_end - x_start < 200 or y_end - y_start < 200:
                    break
                new_img = img[y_start:y_end, x_start:x_end]
                new_filename = filename.split(".")[0] + "_{}.png".format(count)
                new_path = os.path.join(dir_path2, new_filename)
                cv2.imwrite(new_path, new_img)
                count += 1
                if x_end == width and y_end == height:
                    break
                elif x_end == width:
                    x_start = 0
                    y_start = y_end - 200
                else:
                    x_start = x_end - 200
                    y_start = y_start

        # 如果宽和高都小于等于600像素，则将图像保存到输出目录中
        else:
            new_path = os.path.join(dir_path2, filename)
            cv2.imwrite(new_path, img)
