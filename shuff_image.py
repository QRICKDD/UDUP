import os
import shutil
from AllConfig.GConfig import abspath
# 设置文件夹A和文件夹B的路径
folder_a_path = os.path.join(abspath,'AllData/train_clip')
folder_b_path = os.path.join(abspath,'AllData/train')
# dir_path = os.path.join(abspath,'AllData/train')
# dir_path2 = os.path.join(abspath,'AllData/train_clip') # 修改为实际路径
# 获取文件夹A中的所有图片
image_list = [f for f in os.listdir(folder_a_path) if f.endswith('.png')]

# 遍历图片列表，将每个图片复制到文件夹B中并重命名
for i, image_name in enumerate(image_list):
    image_path = os.path.join(folder_a_path, image_name)
    new_image_name = str(i+1).zfill(3) + '.png'  # 生成新的文件名，如001.png
    new_image_path = os.path.join(folder_b_path, new_image_name)
    shutil.copy2(image_path, new_image_path)  # 复制文件到新的路径并重命名
