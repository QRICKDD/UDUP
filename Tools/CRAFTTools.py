from AllConfig import GConfig
import os
import torch
from model_CRAFT.basenet.craft import CRAFT
import numpy as np
import pyclipper
import cv2
from collections import OrderedDict
from Tools.Baseimagetool import *
from model_CRAFT import craft_utils

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
def load_CRAFTmodel():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #加载模型
    CRAFT_path=os.path.join(os.path.abspath('.'),GConfig.CRAFT_model_path)
    #CRAFT_path=GConfig.CRAFT_model_path
    craft_net = CRAFT()
    craft_net.load_state_dict(copyStateDict(torch.load(CRAFT_path)))
    craft_net.to('cuda:0')
    craft_net.eval()
    return craft_net

#square_size=1280
def get_CRAFT_pred(model,img:torch.Tensor,square_size=1280,is_eval=False):
    img_resized, target_ratio =resize_aspect_ratio(image=img,square_size=square_size,
                                                   mag_ratio=1.0)
    #img_norm=normlize_MeanVariance(img_resized,device)
    img_norm=img_resized
    if is_eval:
        with torch.no_grad():
            y,_=model(img_norm)
    else:
        y,_=model(img_norm)

    score_text = y[0, :, :, 0]
    score_link = y[0, :, :, 1]
    return (score_text,score_link,target_ratio)

def get_CRAFT_box(score_text,score_link,target_ratio,
                  text_threshold=0.7,link_threshold=0.4,low_text=0.4):
    score_link=score_link.clone().detach().cpu().data.numpy()
    score_text = score_text.clone().detach().cpu().data.numpy()
    ratio_h = ratio_w = 1 / target_ratio
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold,
                                           link_threshold, low_text, poly=True)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    # for k in range(len(polys)):
    #     if polys[k] is None: polys[k] = boxes[k]
    return boxes#,polys

#这里的img就是CV2读出来的
def CRAFT_draw_box(img,boxes,save_path=r"..\result_save\test_save\craft_boxes.jpg"):
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(save_path, img)

