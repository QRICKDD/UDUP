import os
from PIL import Image

# 输入文件夹路径
input_folder = r"C:\Users\djc\Desktop\论文集合\OCRUDUP-TIFS\IEEE-Transactions-LaTeX2e-templates-and-instructions\tfs\30"

# 输出文件夹路径
output_folder = r"C:\Users\djc\Desktop\论文集合\OCRUDUP-TIFS\IEEE-Transactions-LaTeX2e-templates-and-instructions\tfs\30-jepg"

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 如果文件是png格式
    if filename.endswith(".png"):
        # 拼接文件路径
        filepath = os.path.join(input_folder, filename)
        # 打开图片
        with Image.open(filepath) as img:
            # 将png图片转换为JPEG格式
            img = img.convert("RGB")
            # 拼接输出文件路径
            output_filepath = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
            # 保存图片
            img.save(output_filepath)
