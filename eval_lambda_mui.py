import torch
import cv2
import os
from torchvision import utils as vutils

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

import warnings
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()

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

def evaluate_and_draw(model,adv_patch, image_root, gt_root,pan_cfg,
                      save_path,model_name):
    global evaluator
    image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
    images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
    test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]
    results = []  # PRF


    for img, name, gt in zip(images, image_names, test_gts):
        print(name)
        h, w = img.shape[2:]

        UAU = repeat_4D(adv_patch.clone().detach(), h, w)
        mask_t = extract_background(img).cuda()
        img = img.cuda()
        merge_image = img * (1 - mask_t) + mask_t * UAU
        merge_image = merge_image.cuda()
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

        gt = read_txt(gt)
        temp = evaluator.evaluate_image(gt, boxes)
        results.append(temp)


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

def main(patch_path,img_path,gt_path,save_path,model_name):
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
    adv_patch=torch.load(patch_path)
    adv_patch=adv_patch.cuda()
    P, R, F=evaluate_and_draw(model=model,adv_patch=adv_patch,image_root=img_path,
                              gt_root=gt_path, save_path=save_path,pan_cfg=pan_cfg,
                              model_name=model_name)
    del model
    return R, P, F,torch.mean(1-adv_patch)


def gen_iter_dict(special_eval_item):
    mypath=os.path.join(abspath,'result_save_0.06',special_eval_item)
    dir_list=os.listdir(mypath)
    is_need=[]
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need

import logging
from Tools.Log import logger_config
if __name__=="__main__":

    root_eval = os.path.join(abspath, 'result_save_0.06')
    img_path = os.path.join(abspath, "AllData", "test")
    # special_eval=[
    #     "size=10_step=3_eps=120_lambdaw=0.001",
    #     "size=20_step=3_eps=120_lambdaw=0.1",
    #     "size=30_step=3_eps=120_lambdaw=0.1",
    #     "size=50_step=3_eps=120_lambdaw=0.001",
    #     "size=100_step=3_eps=120_lambdaw=0.1",
    #     "size=150_step=3_eps=120_lambdaw=0.1",
    #     "size=200_step=3_eps=120_lambdaw=0.01"
    # ]
    special_eval=[
        "size=10_step=3_eps=120_lambdaw=0",
        "size=20_step=3_eps=120_lambdaw=0",
        "size=30_step=3_eps=120_lambdaw=0",
        "size=50_step=3_eps=120_lambdaw=0",
        "size=100_step=3_eps=120_lambdaw=0",
        "size=150_step=3_eps=120_lambdaw=0",
        "size=200_step=3_eps=120_lambdaw=0"
    ]

    muis=[0.09,0.12]
    mui_filters=[]
    filter=[]

    for dir in special_eval:#迭代
        #log save path
        sele_size=(dir.split('=')[1]).split('_')[0]
        if int(sele_size) in filter:
            continue
        log_path = os.path.join(abspath, 'Mylog_0.06\DiffSize_abs')
        log_name = os.path.join(abspath, 'Mylog_0.06\DiffSize_abs\{}.log'.format(sele_size))
        #clean logger
        logger = logger_config(log_path=log_path,log_filename=log_name)
        while len(logger.handlers)!=0:
            logger.removeHandler(logger.handlers[0])
        logger = logger_config(log_filename=log_name)
        logger.info("lambda={}".format(dir.split('=')[-1]))
        p1 = os.path.join(root_eval,dir)
        iter_eval=gen_iter_dict(dir)
        iter_eval=sorted(iter_eval,key=lambda  x:int(x))
        for mui,iter in zip(muis,iter_eval):
            if mui in mui_filters:
                continue
            logger.info("iter={},mui={}".format(iter,mui))
            patch_path = os.path.join(p1,"advtorch","advpatch_{}".format(iter))

            for model_name in ["easyocr","psenet","dbnet","panpp","panet","craft"]:
                gt_path=os.path.join(abspath,"AllData","test_"+model_name+"_gt")
                save_path = os.path.join(abspath,"test_save", dir,model_name,str(iter)+'_'+str(mui))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                R,P,F,pert = main(patch_path, img_path, gt_path, save_path, model_name)
                logger.info("model:{},iter:{} mui:{} R:{:.4f} P:{:.4f} F:{:.4f} MUI:{:.4f}".
                        format(model_name,iter,mui, R, P, F,pert))