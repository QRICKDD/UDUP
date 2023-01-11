import cv2
import os
gt_dir_path=r"../AllData/train_gt"
image_dir_path=r"../AllData/train"
all_gt_names=os.listdir(gt_dir_path)#名字
all_gt_path=[os.path.join(gt_dir_path,item) for item in all_gt_names]#绝对路径

def get_img_via_txt(txt_name):
    return os.path.join(image_dir_path,txt_name.split('_')[0]+".png")


from Tools.EvalTool import get_mask
from Tools.Showtool import img_show1
save_path=r"../AllData/train_mask"
for txt_name,txt_path in zip(all_gt_names,all_gt_path):
    img=cv2.imread(get_img_via_txt(txt_name))
    [h,w]=img.shape[:2]
    mask=get_mask(txt_path,[h,w])
    #print(mask)
    #img_show1(mask)
    cv2.imwrite(os.path.join(save_path,txt_name.split('_')[0]+".png"),mask*255)



