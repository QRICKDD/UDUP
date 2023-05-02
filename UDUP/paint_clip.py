from Tools.ImageIO import img_read,img_tensortocv2
import cv2
from Tools.Showtool import img_show3_t
dir_path=r"F:\udup\temp-save"
import os
all_f=os.listdir(dir_path)
all_res=[]
all_name=[]
for item in all_f:
    if item.split('.')[-1]!="png":
        continue
    else:
        all_res.append(os.path.join(dir_path,item))
        all_name.append(item)
for item,name in zip(all_res,all_name):
    img=img_read(item)
    img=img[:,:,100:400,200:500]
    cv2.imwrite(os.path.join(dir_path,"CP_"+name),img_tensortocv2(img))





