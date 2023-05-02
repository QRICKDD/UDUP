from Tools.Baseimagetool import *
from Tools.ImageIO import *
from Tools.Showtool import img_show3_t
from AllConfig.GConfig import *
import cv2
import os
import torch


save_path=os.path.join(abspath,"AliShow")
if os.path.exists(save_path)==False:
    os.makedirs(save_path)


def extract_background2(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_sum = torch.sum(img_tensor, dim=1)
    mask = (img_sum >2.8)
    mask = mask + 0
    return mask.unsqueeze_(0)


adv_path=r"K:\udup_djc\result_save_ali\size=30_step=3_eps=120_lambdaw=0.1\advtorch\advpatch_101"



show_dir=os.path.join(abspath,"AllData/Ali")
all_show_image=os.listdir(show_dir)
for image in all_show_image:
    img=img_read(os.path.join(show_dir,image))
    h, w = img.shape[2:]
    adv_patch=torch.load(adv_path)
    UAU = repeat_4D(adv_patch.clone().detach(), h, w).cuda()
    mask_t = extract_background2(img).cuda()
    img = img.cuda()
    merge_image = img * (1 - mask_t) + mask_t * UAU
    merge_image = merge_image.cuda()
    cv2.imwrite(os.path.join(save_path,image),img_tensortocv2(merge_image))