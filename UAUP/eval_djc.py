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
import warnings
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()

def dict_div_num(input_dict:dict,div_num:int):
    for key,value in input_dict.items():
        input_dict[key]["precision"]=input_dict[key]["precision"]/div_num
        input_dict[key]["recall"] =input_dict[key]["recall"]  / div_num
        input_dict[key]["hmean"] = input_dict[key]["hmean"] / div_num
    return input_dict

def evaluate_and_draw(model,adv_patch, image_root, gt_root,
                      save_path, test_type,model_name="CRAFT"):
    global evaluator
    image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
    images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
    test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]
    results = []  # PRF

    color_dict = {"black":{"precision":0,"recall":0,"hmean":0},
                  "blue":{"precision":0,"recall":0,"hmean":0},
                  "gray":{"precision":0,"recall":0,"hmean":0},
                  "green":{"precision":0,"recall":0,"hmean":0},
                  "yellow":{"precision":0,"recall":0,"hmean":0},
                  "red":{"precision":0,"recall":0,"hmean":0}}

    font_dict = {"8":{"precision":0,"recall":0,"hmean":0},
                 "9":{"precision":0,"recall":0,"hmean":0},
                 "10":{"precision":0,"recall":0,"hmean":0},
                 "11":{"precision":0,"recall":0,"hmean":0},
                 "12":{"precision":0,"recall":0,"hmean":0},
                 "14":{"precision":0,"recall":0,"hmean":0},
                 "16":{"precision":0,"recall":0,"hmean":0},
                 "18":{"precision":0,"recall":0,"hmean":0},
                 "20":{"precision":0,"recall":0,"hmean":0}}

    for img, name, gt in zip(images, image_names, test_gts):
        h, w = img.shape[2:]

        UAU = repeat_4D(adv_patch.clone().detach(), h, w)
        mask_t = extract_background(img).cuda()
        img = img.cuda()
        merge_image = img * (1 - mask_t) + mask_t * UAU
        if test_type.split("_")[0] == "resize":
            scale_factor = int(test_type.split("_")[1]) * 0.01
            merge_image = torch.tensor(transforms.Resize([int(h * scale_factor),
                                                              int(w * scale_factor)])(merge_image))

        merge_image = merge_image.to('cuda:0')

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
        temp = evaluator.evaluate_image(gt, boxes)
        results.append(temp)

        if test_type == "font":
            n1 = name.split('\\')[-1].split('.')[0].split('_')[0]
            font_dict[n1]["precision"] += temp["precision"]
            font_dict[n1]["recall"] += temp["recall"]
            font_dict[n1]["hmean"] += temp["hmean"]

        if test_type == "color":
            n1 = name.split('\\')[-1].split('.')[0].split('_')[0]
            color_dict[n1]["precision"] += temp["precision"]
            color_dict[n1]["recall"] += temp["recall"]
            color_dict[n1]["hmean"] += temp["hmean"]

        temp_save_path = os.path.join(save_path, name.split('\\')[-1])
        Draw_box(img_tensortocv2(merge_image), boxes, temp_save_path, model_name=model_name)
    color_dict=dict_div_num(color_dict,10)
    font_dict = dict_div_num(font_dict, 2)
    P, R, F = evaluator.combine_results(results)
    return P, R, F ,color_dict,font_dict

def init_dataset(test_path,img_path,gt_path):
    train_dataset=[]
    test_img_path=os.path.join(test_path,img_path)
    test_gt_path = os.path.join(test_path, gt_path)
    test_images = [img_read(os.path.join(test_img_path, name)) for name in os.listdir(test_img_path)]
    test_gts = [os.path.join(test_gt_path,name) for name in os.listdir(test_gt_path)]
    for image,gt in zip(test_images,test_gts):
        train_dataset.append([image,gt])
    return train_dataset

def main(patch_path,img_path,gt_path,save_path,model_name,test_type):
    if model_name=='CRAFT':
        model = load_CRAFTmodel()
    else:
        model = load_DBmodel()
    adv_patch=torch.load(patch_path)
    adv_patch=adv_patch.cuda()
    P, R, F,color_dict,font_dict = evaluate_and_draw(model,adv_patch, img_path,gt_path, save_path,model_name=model_name,test_type = test_type)
    if test_type == "font":
        return torch.mean(1 - adv_patch), font_dict

    if test_type == "color":
        return torch.mean(1-adv_patch),color_dict
    return R, P, F,torch.mean(1-adv_patch)

def logINFO(log,f):

    print(log,file=f,flush=True)
    print(log)

#----------0.06 0.07 0.08 0.09 0.1
iter_dict={10:[18,21,24,26,29],
           20:[22,25,28,31,34],
           30:[21,25,29,32,35],
           50:[22,27,30,33,37],
           80:[24,27,30,33,37],
           100:[23,27,30,33,38],
           150:[24,28,32,36,41],
           200:[27,30,34,39,45]}

if __name__=="__main__":

    root_eval = r'F:\udup\result_save'
    test_types = ["color"]
    model_name = "CRAFT"

    for dir in os.listdir(root_eval):
        log_path = r'F:\udup\Mylog\eval_log\{}'.format(dir)
        log_file = open(log_path, 'w')
        p1 = os.path.join(root_eval,dir)
        sele_size=int(dir.split('_')[0][5:])
        iter_eval=iter_dict[sele_size]
        for iter in iter_eval:
            patch_path = os.path.join(p1,"advtorch","advpatch_{}".format(iter))
            for test_type in test_types:
                if test_type!='test':
                    temp_gt_path = os.path.join(os.path.abspath('.')[:-4], "AllData",
                                             "test_{}_gt".format(test_type))
                    img_path = os.path.join(os.path.abspath('.')[:-4], "AllData",
                                            "test_{}".format(test_type))
                else:
                    temp_gt_path = os.path.join(os.path.abspath('.')[:-4], "AllData","test_gt")
                    img_path = os.path.join(os.path.abspath('.')[:-4], "AllData","test")
                if test_type == "resize":
                    for d1 in os.listdir(temp_gt_path):
                        p2 = os.path.join(temp_gt_path,d1)
                        save_path = os.path.join(p1, str(iter),'test_{}'.format(test_type),d1)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        gt_path = os.path.join(os.path.abspath('.')[:-4], "AllData","test_{}_gt".format(test_type),d1)
                        R,P,F,pert = main(patch_path, img_path, gt_path, save_path, model_name,test_type = test_type + "_" + d1)
                        logINFO("test_type:{} resize: {} iter:{} R:{:.4f} P:{:.4f} F:{:.4f} MUI:{:.4f}".format(test_type,d1,iter,R,P,F,pert), log_file)

                elif test_type=="color" or test_type=='font':
                    gt_path = temp_gt_path
                    save_path = os.path.join(p1, str(iter), 'test_{}'.format(test_type))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    pert,dict = main(patch_path,img_path,gt_path,save_path,model_name,test_type = test_type)
                    for k,v in dict.items():
                        logINFO("test_type:{}-{} resize: {} iter:{} R:{:.4f} P:{:.4f} F:{:.4f} MUI:{:.4f}".
                                format(test_type,k,100,iter,dict[k]["recall"],dict[k]["precision"],dict[k]["hmean"],pert), log_file)
                elif test_type=='test':
                    gt_path = temp_gt_path
                    save_path = os.path.join(p1, str(iter), 'original')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    R,P,F,pert = main(patch_path, img_path, gt_path, save_path, model_name, test_type=test_type)
                    logINFO("test_type:{} iter:{} R:{:.4f} P:{:.4f} F:{:.4f} MUI:{:.4f}".format(test_type,
                                                                                                       iter, R, P, F,
                                                                                                       pert), log_file)







