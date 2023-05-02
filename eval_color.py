import torch
import cv2
import os
from torchvision import utils as vutils
import warnings

from model_CRAFT.pred_single import *
from model_PAN.pred_single import pan_pred_single
from mmocr.utils.ocr import MMOCR
import easyocr
from Tools.ImageIO import *
from Tools.Baseimagetool import *
from Tools.EvalTool import *
from Tools.CRAFTTools import *
from Tools.DBTools import *
from Tools.PANTools import *

from UDUP.Auxiliary import *

warnings.filterwarnings("ignore")



def get_result_easyocr(result):
    points=[]
    for item in result:
        points.append(item[0])
    return np.array(points)

def dict_div_num(input_dict:dict,div_num:int):
    for key,value in input_dict.items():
        input_dict[key]["precision"]=input_dict[key]["precision"]/div_num
        input_dict[key]["recall"] =input_dict[key]["recall"]  / div_num
        input_dict[key]["hmean"] = input_dict[key]["hmean"] / div_num
    return input_dict

def evaluate_and_draw(model,adv_patch, image_root,pan_cfg,
                      save_path,model_name):
    evaluator=DetectionIoUEvaluator()
    names_color=["black","blue","gray","green","red","yellow"]
    image_names = [item+"_007.png" for item in names_color]+\
                  [item+"_005.png" for item in names_color]+[item+"_001.png" for item in names_color]+\
                  [item+"_009.png" for item in names_color]+[item+"_002.png" for item in names_color]+\
                  [item+"_004.png" for item in names_color]+[item+"_003.png" for item in names_color]+\
                  [item+"_008.png" for item in names_color]

    images = [img_read(os.path.join(image_root,item+"_007.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_005.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_001.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_009.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_002.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_004.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_003.png")) for item in names_color]+\
             [img_read(os.path.join(image_root,item+"_008.png")) for item in names_color]
    results = []  # PRF


    for img, name in zip(images, image_names):
        print(name)
        h, w = img.shape[2:]
        if adv_patch!=None:
            UAU = repeat_4D(adv_patch.clone().detach(), h, w)
            mask_t = extract_background(img).cuda()
            img = img.cuda()
            merge_image = img * (1 - mask_t) + mask_t * UAU
            merge_image = merge_image.cuda()
        else:
            merge_image=img
        if model_name == 'dbnet':
            array_img = img_tensortocv2(merge_image)
            boxes = get_pred_boxes_formmocr(model.readtext2(array_img))
            boxes = recover_mmocr_boxes(array_img, boxes, model_name=model_name)

        elif model_name == 'craft':
            model = load_CRAFTmodel()
            (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [],
                                                                           model_name,is_eval=True)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                                  text_threshold=0.7, link_threshold=0.4, low_text=0.4)

        elif model_name=="easyocr":
            vutils.save_image(merge_image,os.path.join(abspath,'AllData/temp/temp.png'))
            result = model.readtext(os.path.join(abspath,'AllData/temp/temp.png'))
            boxes = get_result_easyocr(result)

        elif model_name=="panpp":
            data=pan_preprocess_image(merge_image)#dict(imgs=img, img_metas=img_meta)
            boxes=pan_pred_single(model,pan_cfg,data)#需要修改
            boxes = get_pred_boxes_forpanpp(boxes)

        elif model_name=="panet":
            array_img=img_tensortocv2(merge_image)
            boxes = get_pred_boxes_formmocr(model.readtext2(array_img))
            boxes=recover_mmocr_boxes(array_img, boxes, model_name=model_name)

        elif model_name=="psenet":
            array_img = img_tensortocv2(merge_image)
            boxes = get_pred_boxes_formmocr(model.readtext2(array_img))
            boxes = recover_mmocr_boxes(array_img, boxes, model_name=model_name)


        temp_save_path = os.path.join(save_path, name.split('\\')[-1])
        Draw_box(img_tensortocv2(merge_image), boxes, temp_save_path, model_name=model_name)
    P, R, F = evaluator.combine_results(results)
    return P, R, F

def init_dataset(test_path,img_path,gt_path):
    train_dataset=[]
    test_img_path=os.path.join(test_path,img_path)
    test_gt_path = os.path.join(test_path, gt_path)
    test_images = [img_read(os.path.join(test_img_path, name)) for name in os.listdir(test_img_path)]
    test_gts = [os.path.join(test_gt_path,name) for name in os.listdir(test_gt_path)]
    for image,gt in zip(test_images,test_gts):
        train_dataset.append([image,gt])
    return train_dataset

def main(patch_path,img_path,save_path,model_name):
    pan_cfg=None
    if model_name=="craft":
        model = load_CRAFTmodel()
    elif model_name=='dbnet':
        model = MMOCR(det='DB_r18', recog=None)
    elif model_name=='panet':
        model = MMOCR(det='PANet_IC15', recog=None)
    elif model_name=='psenet':
        model=MMOCR(det='PS_IC15',recog=None)
    elif model_name=='easyocr':
        model = easyocr.Reader(['en'], gpu=True,
                                model_storage_directory=os.path.join(abspath, r'AllConfig/all_model'))
    elif model_name=="panpp":
        model,pan_cfg=load_PANPlusmodel()
    if patch_path!=None:
        adv_patch=torch.load(patch_path)
        adv_patch=adv_patch.cuda()
    else:
        adv_patch=None
    P, R, F=evaluate_and_draw(model=model,adv_patch=adv_patch,image_root=img_path,
                              save_path=save_path,pan_cfg=pan_cfg,
                              model_name=model_name)
    del model
    return R, P, F,torch.mean(1-adv_patch)



import logging
from Tools.Log import logger_config
if __name__=="__main__":

    root_eval = os.path.join(abspath, 'result_save_0.06')
    img_path = os.path.join(abspath, "AllData", "test_color")
    special_eval=[
        None,
        #"size=10_step=3_eps=120_lambdaw=0.001/advtorch/advpatch_24",
        #"size=20_step=3_eps=120_lambdaw=0.1/advtorch/advpatch_24"#27,
        #"size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch_26",#29
        #"size=50_step=3_eps=120_lambdaw=0.001/advtorch/advpatch_26",#29
        #"size=100_step=3_eps=120_lambdaw=0.1/advtorch/advpatch_28"#31,
        #"size=150_step=3_eps=120_lambdaw=0.1/advtorch/advpatch_33",#33
        #"size=200_step=3_eps=120_lambdaw=0.01/advtorch/advpatch_32"#35
    ]

    for adv_patch in special_eval:#迭代
        if adv_patch!=None:
            sele_size = (adv_patch.split('=')[1]).split('_')[0]
            adv_patch = os.path.join(root_eval,adv_patch)


            save_path = os.path.join(abspath,"color_save",str(sele_size))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            main(adv_patch, img_path, save_path, 'craft')
        else:
            save_path = os.path.join(abspath, "color_save", str("None"))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            main(None, img_path, save_path, 'craft')