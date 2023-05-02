import matplotlib.pyplot as plt
import os
from AllConfig.GConfig import abspath
#img_path=os.path.join(abspath,"AllData/test/010.png")
img_path=r"C:\Users\djc\Documents\GitHub\udup_djc\AllData\test\016.png"
txt_path=os.path.join(abspath,"commercialOCR/baidu.txt")
txt_path=r"C:\Users\djc\Documents\GitHub\udup_djc\commercialOCR/baidu.txt"
import cv2
from Tools.ImageIO import *
img=img_read(img_path)
img_h,img_w=img.shape[2:]
import torch
def init_white(img_h=img_h,img_w=img_w):
    global img
    img=torch.ones([1,3,img_h,int(img_w)])
    return img_tensortocv2(img)
img=init_white()

with open(txt_path,"r",encoding='utf-8') as f:
    lines=f.readlines()
content=[]
hs,ws=[],[]
xs,ys=[],[]
opena,openb,openc=True,False,False
for idx,item in enumerate(lines):
    item=item.strip()
    if opena:
        cts=" ".join(item.split(' ')[1:])
        content.append(cts)
        opena=False
        openb=True
        openc=False
    elif openb:
        w=int((item.split(r"宽度：")[-1].split(r"高度：")[0]))
        h = int(item.split("高度：")[-1])
        ws.append(w)
        hs.append(h)
        opena = False
        openb = False
        openc = True
    elif openc:
        x=int((item.split(r"左间距：")[-1]).split(r"上间距：")[0])
        y = int(item.split(r"上间距：")[-1])
        xs.append(x)
        ys.append(y)
        opena = True
        openb = False
        openc = False

for con,h,w,x,y in zip(content,hs,ws,xs,ys):
    cv2.putText(
        img,  # 图像
        con,  # 文字
        (x,y+h),  # 文字左下角，(w_idx, h_idx)
        cv2.FONT_HERSHEY_SIMPLEX,  # 字体
        1.8,  # 字体大小
        (255, 0,0),  # 字体颜色
        2,  # 线宽 单位是像素值
        cv2.LINE_AA  # 线的类型
    )


cv2.imwrite(r"C:\Users\djc\Documents\GitHub\udup_djc\commercialOCR\save.png",img)
