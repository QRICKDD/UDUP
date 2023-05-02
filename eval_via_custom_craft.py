import torch
import cv2
import os
from torchvision import utils as vutils
import warnings
from model_CRAFT.pred_single import *
from Tools.ImageIO import *
from Tools.Baseimagetool import *
from Tools.EvalTool import *
from Tools.CRAFTTools import *
from AllConfig.GConfig import abspath
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

def evaluate_and_draw(model,adv_patch, image_root, gt_root,
                      save_path):
    evaluator=DetectionIoUEvaluator()
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

        model = model
        (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [],
                                                                       model_name='craft',is_eval=True)
        boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                              text_threshold=0.7, link_threshold=0.4, low_text=0.4)

        gt = read_txt(gt)
        temp = evaluator.evaluate_image(gt, boxes)
        results.append(temp)


        temp_save_path = os.path.join(save_path, name.split('\\')[-1])
        Draw_box(img_tensortocv2(merge_image), boxes, temp_save_path,'craft')
    P, R, F = evaluator.combine_results(results)
    del(evaluator)
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

def main(patch_path,img_path,gt_path,save_path):
    pan_cfg=None
    model = load_CRAFTmodel()
    adv_patch=torch.load(patch_path)
    adv_patch=adv_patch.cuda()
    P, R, F=evaluate_and_draw(model=model,adv_patch=adv_patch,image_root=img_path,
                              gt_root=gt_path, save_path=save_path)
    del model
    return R, P, F,torch.mean(1-adv_patch)


import logging
import os
from Tools.Log import logger_config
if __name__=="__main__":
    img_test_path=os.path.join(abspath,"AllData/test")
    root_eval = os.path.join(abspath, 'result_save_0.06')
    result_dir=[
        "size=10_step=3_eps=120_lambdaw=0.001",
        "size=20_step=3_eps=120_lambdaw=0.1",
        "size=30_step=3_eps=120_lambdaw=0.1",
        "size=50_step=3_eps=120_lambdaw=0.001",
        "size=100_step=3_eps=120_lambdaw=0.1",
        "size=150_step=3_eps=120_lambdaw=0",
        "size=200_step=3_eps=120_lambdaw=0.01",
    ]
    #0.06 0.07 0.08 0.1 0.11
    bake_dict=[
        ["18","20","22","27","29"],#10
        ["20","23","25","30","32"],#20
        ["21","24","26","32","35"],#30
        ["20","23","26","32","35"],#50
        ["21","25","28","34","38"],#100
        ["23","26","30","37","41"],#150
        ["24","28","31","40","44"],#200
    ]
    muis=[0.06,0.07,0.08,0.10,0.11]
    for dir,iter_lists in zip(result_dir,bake_dict):#迭代
        #log save path
        sele_size=(dir.split('=')[1]).split('_')[0]
        log_path = os.path.join(abspath, 'Mylog_0.06')
        log_name = os.path.join(abspath, 'Mylog_0.06\{}.log'.format(dir))
        #clean logger
        logger = logger_config(log_path=log_path,log_filename=log_name)
        while len(logger.handlers)!=0:
            logger.removeHandler(logger.handlers[0])
        logger = logger_config(log_filename=log_name)
        logger.info("lambda={}".format(dir.split('=')[-1]))

        result_save_path = os.path.join(root_eval,dir)#result_0.06/size=20_step=3_eps=120_lambdaw=0.1
        for mui,iter in zip(muis,iter_lists):
            logger.info("iter={},mui={}".format(iter,mui))
            patch_path = os.path.join(result_save_path,"advtorch","advpatch_{}".format(iter))
            gt_path=os.path.join(abspath,"AllData","test_craft_gt")

            save_path = os.path.join(result_save_path,str(mui))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            R,P,F,pert = main(patch_path, img_test_path, gt_path, save_path)
            logger.info("iter:{} mui:{} R:{:.4f} P:{:.4f} F:{:.4f} MUI:{:.4f}".
                    format(iter,mui, R, P, F,pert))