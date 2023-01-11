import torch
from model_CRAFT.pred_single import *
from Tools.ImageIO import *
from Tools.Baseimagetool import *
import os
from Tools.CRAFTTools import *
from Tools.DBTools import *
import cv2
from UAUP.Auxiliary import *
from Tools.EvalTool import *

evaluator=DetectionIoUEvaluator()

def evaluate_and_draw(model,adv_patch, image_root, gt_root,
                      save_path, resize_ratio=0, is_resize=False,
                      model_name="CRAFT"):
    global evaluator
    image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
    images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
    test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]
    results = []  # PRF
    for img, name, gt in zip(images, image_names, test_gts):
        h, w = img.shape[2:]
        UAU = repeat_4D(adv_patch.clone().detach(), h, w)
        mask_t = extract_background(img)
        merge_image = img * (1 - mask_t) + mask_t * UAU
        merge_image = merge_image.to('cuda:0')
        if is_resize:
            merge_image = random_image_resize(merge_image, low=resize_ratio, high=resize_ratio)
            h, w = merge_image.shape[2:]
        if model_name == 'DBnet':
            preds = single_grad_inference(model, merge_image, [], model_name)
            preds = preds[0]
            dilates, boxes = get_DB_dilateds_boxes(preds, h, w, min_area=100)
        elif model_name == 'CRAFT':
            (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [],
                                                                           model_name)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                                  text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        gt = read_txt(gt)
        results.append(evaluator.evaluate_image(gt, boxes))
        # draw
        # cv2_img=cv2.imread(name)
        temp_save_path = os.path.join(save_path, name.split('\\')[-1])
        # Draw_box(cv2_img,results,save_path)
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
    if model_name=='CRAFT':
        model = load_CRAFTmodel()
    else:
        model = load_DBmodel()
    adv_patch=torch.load(patch_path)
    adv_patch=adv_patch.cuda()
    P, R, F = evaluate_and_draw(model,adv_patch, img_path,gt_path, save_path,model_name=model_name)
    print("R:{},P:{},F:{}".format(R,P,F))

if __name__=="__main__":
    patch_path=os.path.join(os.path.abspath('.')[:-4],"result_save",
                            "size=100_step=2_eps=70_PI=False",
                            "advtorch","advpatch_60")
    model_name="CRAFT"
    img_path=os.path.join(os.path.abspath('.')[:-4],"AllData",
                          "test_color")
    gt_path = os.path.join(os.path.abspath('.')[:-4], "AllData",
                            "test_color_gt")
    save_path = os.path.join(os.path.abspath('.')[:-4],
                           'result_save','test_color')
    # print(gt_path)
    main(patch_path,img_path,gt_path,save_path,model_name)





