import torch

from Tools.Baseimagetool import *
import cv2
from Tools.ImageIO import *
import os
from torchvision import transforms
from Tools.CRAFTTools import *
import numpy as np
def get_image_hw(image_list):
    hw_list = []
    for item in image_list:
        hw_list.append(item.shape[2:])
    return hw_list


def get_random_resize_image(adv_image_lists, low=0.25, high=3.0):
    resize_adv_img_lists = []
    for img in adv_image_lists:
        resize_adv_img_lists.append(random_image_resize(img, low, high))
    return resize_adv_img_lists


def get_random_resize_image_single(adv_image, low=0.25, high=3.0):
    return random_image_resize(adv_image, low, high)


def get_random_noised_image(adv_image):
    return random_noise(adv_image)


def get_augm_image(adv_images):
    resize_images = get_random_resize_image(adv_images)
    return adv_images + resize_images


def save_adv_patch_img(img_tensor, path):
    img_cv = img_tensortocv2(img_tensor)
    cv2.imwrite(path, img_cv)


def init_test_dataset(data_root):
    test_dataset=[]
    test_path = os.path.join(data_root, "test")
    test_gt_path = os.path.join(data_root, "test_craft_gt")
    test_images = [img_read(os.path.join(test_path, name)) for name in os.listdir(test_path)]
    test_gts=[os.path.join(test_gt_path,name) for name in os.listdir(test_gt_path)]
    for image,gt in zip(test_images,test_gts):
        test_dataset.append([image,gt])
    return test_dataset

def init_train_dataset(data_root):
    train_dataset=[]
    train_path = os.path.join(data_root, "train")
    train_images = [img_read(os.path.join(train_path, name)) for name in os.listdir(train_path)]
    for image in train_images:
        train_dataset.append([image])
    return train_dataset

def Diverse_module_1(image,now_ti,gap):
    high_index=1.08
    low_index = 0.95
    max_resize_range = (0.80, 1.5)
    pow_num=now_ti//gap
    now_resize_low=max(pow(low_index,pow_num),max_resize_range[0])
    now_resize_high = min(pow(high_index, pow_num),max_resize_range[1])
    resize_image=random_image_resize(image,low=now_resize_low,high=now_resize_high)
    h,w=image.shape[2:]
    # resize_mask = transforms.Resize([h, w])(mask)
    return resize_image


def Diverse_module_2(image,UAU,now_ti,gap):
    resize_image=Diverse_module_1(image,now_ti,gap)
    noise_max=0.1
    noise_start=0.01
    noise_index=1.5
    pow_num=now_ti//gap
    now_noise=min(pow(noise_index,pow_num)*noise_start,noise_max)
    noise_resize_image=random_noise(resize_image,-1*now_noise,now_noise)
    h,w=resize_image.shape[2:]
    resize_UAU=transforms.Resize([h, w])(UAU)
    #extract 0/1 mask where 1 is text region
    #resize_mask=extract_background(resize_mask)
    return noise_resize_image,resize_UAU


# def get_DB_single_result(self, aug_image):
#     preds = self.DBmodel(aug_image)[0]
#     prob_map = preds[0]
#     return prob_map
#
# def get_DB_single_loss(self, res, device):
#     target_prob_map = torch.zeros_like(res)
#     target_prob_map = target_prob_map.to(device)
#     cost = -self.loss(res, target_prob_map)
#     return cost
#
# def get_DB_grad(self, adv_image, adv_patch):
#     db_result = self.get_DB_single_result(adv_image)
#     db_single_loss = self.get_DB_single_loss(db_result, device=GConfig.DB_device)
#     grad_db = torch.autograd.grad(db_single_loss, adv_patch,
#                                   retain_graph=False, create_graph=False)[0]
#     return db_single_loss.detach().cpu().item(), grad_db.detach().cpu()

def recover_mmocr_boxes(img,boxes,model_name):
    if isinstance(img,np.ndarray):
        h,w=img.shape[:2]
    else:
        print("error")
        return
    ratio=1.0
    if model_name=='dbnet':
        scale=(1333, 736)
    elif model_name=='psenet':
        scale=(2240, 2240)
        ratio = 4.0
    elif model_name=='panet':
        scale=(1333, 736)
        ratio=4.0
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    scale_factor=1/scale_factor
    boxes=(boxes*ratio*scale_factor).astype(int)
    if len(boxes)==0:
        return boxes
    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)
    return boxes


def get_pred_boxes_formmocr(boxes,score=0.5):
    """
    score ->转为mmocr使用
    """
    res=[]
    pred=boxes[0]['boundary_result']
    for index,item in enumerate(pred):
        if item[-1]>score:
            res.append([[int(item[0]),int(item[1])],[int(item[2]),int(item[3])],
                        [int(item[4]),int(item[5])],[int(item[6]),int(item[7])]])
    return np.array(res)

def get_pred_boxes_forpanpp(boxes):
    """
    score ->转为mmocr使用
    """
    res=[]
    for index,item in enumerate(boxes):
        res.append([[int(item[0]),int(item[1])],[int(item[2]),int(item[3])],
                    [int(item[4]),int(item[5])],[int(item[6]),int(item[7])]])
    return np.array(res)


class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out

def get_feas_by_hook(model,hook_name):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = []
    for n, m in model.named_modules():
        if n in hook_name:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks

def modified_model_inference(model,img,model_name,is_eval):
    if model_name=='DBnet':
        score_text = model(img)
    elif model_name=='CRAFT' or model_name=="craft":
        score_text = get_CRAFT_pred(model, img=img,square_size=1280, is_eval=is_eval,)
    return score_text

def single_grad_inference(model,img ,feaName,model_name,is_eval):

    if feaName and len(feaName) != 0:
        fea_hooks = get_feas_by_hook(model, hook_name=feaName)
    pp_result = modified_model_inference(model, img,model_name,is_eval)
    if feaName and len(feaName)!=0:
        return pp_result,fea_hooks
    return pp_result