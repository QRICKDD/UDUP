import easyocr
import os
from mmocr.utils.ocr import MMOCR
from UDUP.Auxiliary import *
from Tools.ImageIO import *
from Tools.Baseimagetool import *
from Tools.EvalTool import *
from Tools.CRAFTTools import *
from UDUP.Auxiliary import *
from model_PAN.pred_single import pan_pred_single
from Tools.PANTools import *
import numpy as np
import cv2
img_test_1 = os.path.join(os.path.abspath('.'),"AllData/test/002.png")
img_test_3 = os.path.join(os.path.abspath('.'),"AllData/test/003.png")
img_test_2 = os.path.join(os.path.abspath('.'),"AllData/test/020.png")


def get_result_easyocr(result):
    points=[]
    for item in result:
        points.append(item[0])
    return points

def get_result_mmocr(result):
    points=[]
    for item in result:
        if isinstance(item,(np.ndarray)):
            points.append(item.tolist())
        else:
            points.append(item)
    return points

def write_res(file_path,res):
    with open(file_path,'w') as f:
        for item in res:#这是一组8个[[a,b],[c,d]...]
            ite=[str(int(temp[0]))+","+str(int(temp[1])) for temp in item]
            f.writelines(",".join(ite)+'\n')
    return

def create_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)


# dir_path_for_ijcai=os.path.join(os.path.abspath('.'),"AllData/test_for_ijcai")
# all_images_ijcai=os.listdir(dir_path_for_ijcai)
# all_abs_images_ijcai=[os.path.join(dir_path_for_ijcai,item) for item in all_images_ijcai]
import os
dir_path=os.path.join(os.path.abspath('.'),"AllData/test")
all_images=os.listdir(dir_path)
all_abs_images=[os.path.join(dir_path,item) for item in all_images]

def test_save_easyocr_gt(all_images=all_images,all_abs_images=all_abs_images):
    reader = easyocr.Reader(['en'], gpu=True,
                            model_storage_directory=r'F:\OCR-TASK\OCR__advprotect\AllConfig\all_model')
    easyocr_gt_dir_path=os.path.join(abspath,"AllData/test_easyocr_gt")
    create_dir(easyocr_gt_dir_path)
    for imgname,item in zip(all_images,all_abs_images):
        result = reader.readtext(item)
        res=get_result_easyocr(result)#获取列表
        #目标txt路径=目标路径+图片名称_gt.txt
        target_txt=os.path.join(easyocr_gt_dir_path,imgname.split(".")[0]+"_gt.txt")
        write_res(target_txt,res)

def test_save_dbnet_gt(all_images=all_images,all_abs_images=all_abs_images):
    ocr = MMOCR(det='DB_r18', recog=None)
    dbnet_gt_dir_path = os.path.join(abspath, "AllData/test_dbnet_gt")
    create_dir(dbnet_gt_dir_path)
    for imgname, item in zip(all_images, all_abs_images):
        img = cv2.imread(item)
        boxes = get_pred_boxes_formmocr(ocr.readtext2(img))
        boxes = recover_mmocr_boxes(img, boxes, model_name="dbnet")
        res = get_result_mmocr(boxes)  # 获取列表
        # 目标txt路径=目标路径+图片名称_gt.txt
        target_txt = os.path.join(dbnet_gt_dir_path, imgname.split(".")[0] + "_gt.txt")
        write_res(target_txt, res)

def test_save_panet_gt(all_images=all_images,all_abs_images=all_abs_images):
    ocr = MMOCR(det='PANet_IC15', recog=None)
    pan_gt_dir_path = os.path.join(abspath, "AllData/test_panet_gt")
    create_dir(pan_gt_dir_path)
    for imgname, item in zip(all_images, all_abs_images):
        img = cv2.imread(item)
        boxes = get_pred_boxes_formmocr(ocr.readtext2(img))
        boxes = recover_mmocr_boxes(img, boxes, model_name="panet")
        res = get_result_mmocr(boxes.tolist())  # 获取列表
        # 目标txt路径=目标路径+图片名称_gt.txt
        target_txt = os.path.join(pan_gt_dir_path, imgname.split(".")[0] + "_gt.txt")
        write_res(target_txt, res)

def test_save_psenet_gt(all_images=all_images,all_abs_images=all_abs_images):
    ocr = MMOCR(det='PS_IC15', recog=None)
    pse_gt_dir_path = os.path.join(abspath, "AllData/test_psenet_gt")
    create_dir(pse_gt_dir_path)
    for imgname, item in zip(all_images, all_abs_images):
        img = cv2.imread(item)
        boxes = get_pred_boxes_formmocr(ocr.readtext2(img))
        boxes=recover_mmocr_boxes(img,boxes,model_name="psenet")
        res = get_result_mmocr(boxes.tolist())  # 获取列表
        # 目标txt路径=目标路径+图片名称_gt.txt
        target_txt = os.path.join(pse_gt_dir_path, imgname.split(".")[0] + "_gt.txt")
        write_res(target_txt, res)

def test_save_craft_gt(all_images_ijcai=all_images,all_abs_images_ijcai=all_abs_images):
    model = load_CRAFTmodel()
    craft_gt_dir_path = os.path.join(abspath, "AllData/test_craft_gt")
    create_dir(craft_gt_dir_path)
    for imgname, item in zip(all_images_ijcai, all_abs_images_ijcai):
        img=img_read(item)
        img=img.cuda()
        (score_text, score_link, target_ratio) = single_grad_inference(model, img, [],
                                                                       model_name='craft',is_eval=True)
        boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                              text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        # 目标txt路径=目标路径+图片名称_gt.txt
        target_txt = os.path.join(craft_gt_dir_path, imgname.split(".")[0] + "_gt.txt")
        write_res(target_txt, boxes)

def test_save_panpp_gt(all_images=all_images,all_abs_images=all_abs_images):
    model,pan_cfg=load_PANPlusmodel()
    panpp_gt_dir_path = os.path.join(abspath, "AllData/test_panpp_gt")
    create_dir(panpp_gt_dir_path)
    for imgname, item in zip(all_images, all_abs_images):
        img=img_read(item)
        img=img.cuda()
        data=pan_preprocess_image(img)
        boxes = pan_pred_single(model, pan_cfg, data)
        boxes=get_pred_boxes_forpanpp(boxes)
        # 目标txt路径=目标路径+图片名称_gt.txt
        target_txt = os.path.join(panpp_gt_dir_path, imgname.split(".")[0] + "_gt.txt")
        write_res(target_txt, boxes)

def test_panpp():
    img = img_read(img_test_1)
    model,pan_cfg=load_PANPlusmodel()
    img = img.cuda()
    data = pan_preprocess_image(img)
    boxes = pan_pred_single(model, pan_cfg, data)
    print(boxes)

def test_easyocr():
    reader = easyocr.Reader(['en'], gpu=True,
                                model_storage_directory=r'F:\OCR-TASK\OCR__advprotect\AllConfig\all_model')
    img=cv2.imread(img_test_1)
    # [h,w]=img.shape[1:]
    result = reader.readtext(img_test_1)
    res=get_result_easyocr(result)
    print(res)

def test_db():
    img = cv2.imread(img_test_1)
    ocr = MMOCR(det='DB_r18', recog=None)
    boxes = get_pred_boxes_formmocr(ocr.readtext2(img))
    print(boxes)

def test_panet():
    img = cv2.imread(img_test_2)
    ocr = MMOCR(det='PANet_IC15', recog=None)
    result=ocr.readtext2(img)
    boxes = get_pred_boxes_formmocr(result)
    recover_mmocr_boxes(img,boxes,model_name="panet")
    print(boxes)

def test_craft():
    img = img_read(img_test_1)
    model = load_CRAFTmodel()
    (score_text, score_link, target_ratio) = single_grad_inference(model, img, [],
                                                                   model_name='craft')
    boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                          text_threshold=0.7, link_threshold=0.4, low_text=0.4)
    print(boxes)
