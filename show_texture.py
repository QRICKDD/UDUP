from Tools.Baseimagetool import *
from Tools.ImageIO import *
from Tools.Showtool import img_show3_t
from AllConfig.GConfig import *
import cv2
import os
import torch

show_img=os.path.join(abspath,"AllData/test/012.png")
save_path=os.path.join(abspath,"ImageShow/Experiment1")
if os.path.exists(save_path)==False:
    os.makedirs(save_path)
root_eval = os.path.join(abspath, 'result_save')

def gen_iter_dict(special_eval_item):
    mypath=os.path.join(os.path.join('.'),'result_save',special_eval_item)
    dir_list=os.listdir(mypath)
    is_need=[]
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need

special_eval=[
    "size=10_step=3_eps=120_lambdaw=100",
    "size=20_step=3_eps=120_lambdaw=1",
    "size=30_step=3_eps=120_lambdaw=10",
    "size=50_step=3_eps=120_lambdaw=1",
    "size=100_step=3_eps=120_lambdaw=100",
    "size=150_step=3_eps=120_lambdaw=100",
    "size=200_step=3_eps=120_lambdaw=1",
]
muis=[0.06,0.07,0.08,0.09,0.10,0.11,0.12]
muis_filter=[0.07,0.08,0.10,0.11]



names=[
    "10","20","30","50","100","150","200"
]
for name,dir in zip(names,special_eval):
    p1 = os.path.join(root_eval, dir)
    iter_eval = gen_iter_dict(dir)
    iter_eval = sorted(iter_eval, key=lambda x: int(x))

    for mui,m_iter in zip(muis,iter_eval):
        if mui in muis_filter:
            continue
        img=img_read(show_img)
        h, w = img.shape[2:]
        adv_path=os.path.join(p1,"advtorch","advpatch_{}".format(m_iter))
        adv_patch=torch.load(adv_path)
        UAU = repeat_4D(adv_patch.clone().detach(), h, w).cuda()
        mask_t = extract_background(img).cuda()
        img = img.cuda()
        merge_image = img * (1 - mask_t) + mask_t * UAU
        merge_image = merge_image.cuda()
        img=merge_image[:,:,100:400,200:500]
        cv2.imwrite(os.path.join(save_path,"CP_"+str(mui)+"_"+name+"_019.png"),img_tensortocv2(img))