from Tools.Baseimagetool import *
import cv2
from Tools.ImageIO import *
import os
from torchvision import transforms
from Tools.CRAFTTools import *
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
    test_gt_path = os.path.join(data_root, "test_gt")
    test_images = [img_read(os.path.join(test_path, name)) for name in os.listdir(test_path)]
    test_gts=[os.path.join(test_gt_path,name) for name in os.listdir(test_gt_path)]
    for image,gt in zip(test_images,test_gts):
        test_dataset.append([image,gt])
    return test_dataset

def init_train_dataset(data_root):
    train_dataset=[]
    train_path = os.path.join(data_root, "train")
    train_gt_path = os.path.join(data_root, 'train_gt')
    train_mask_path = os.path.join(data_root, 'train_mask')
    train_images = [img_read(os.path.join(train_path, name)) for name in os.listdir(train_path)]
    train_masks = [img_read(os.path.join(train_mask_path, name)) for name in os.listdir(train_mask_path)]
    train_gts = [os.path.join(data_root,name) for name in os.listdir(train_gt_path)]
    for image,mask,gt_path in zip(train_images,train_masks,train_gts):
        train_dataset.append([image,mask,gt_path])
    return train_dataset

def Diverse_module_1(image,mask,now_ti,gap):
    high_index=1.08
    low_index = 0.95
    max_resize_range = (0.80, 1.5)
    pow_num=now_ti//gap
    now_resize_low=max(pow(low_index,pow_num),max_resize_range[0])
    now_resize_high = min(pow(high_index, pow_num),max_resize_range[1])
    resize_image=random_image_resize(image,low=now_resize_low,high=now_resize_high)
    h,w=image.shape[2:]
    resize_mask = transforms.Resize([h, w])(mask)
    return resize_image,resize_mask


def Diverse_module_2(image,mask,UAU,now_ti,gap):
    resize_image,resize_mask=Diverse_module_1(image,mask,now_ti,gap)
    noise_max=0.1
    noise_start=0.01
    noise_index=1.5
    pow_num=now_ti//gap
    now_noise=min(pow(noise_index,pow_num)*noise_start,noise_max)
    noise_resize_image=random_noise(resize_image,-1*now_noise,now_noise)
    h,w=resize_image.shape[2:]
    resize_UAU=transforms.Resize([h, w])(UAU)
    #extract 0/1 mask where 1 is text region
    resize_mask=extract_background(resize_mask)
    return noise_resize_image,resize_mask,resize_UAU


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

def get_pred_from_boxes(boxes):
    res=[]
    pred=boxes[0]['boundary_result']
    for i in pred:
        res.append([[int(i[0]),int(i[1])],[int(i[2]),int(i[3])],
                    [int(i[4]),int(i[5])],[int(i[6]),int(i[7])]])
    return np.array(res)

class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        ??????????????????feature???hook??????????????????????????????[module, fea_in, fea_out]???????????????????????????????????????????????????
        ?????????????????????????????????torch???????????????module?????????Linear,Conv2d???????????????????????????module????????????????????????
        tuple????????????????????????module????????????????????????tensor????????????????????????????????????????????????????????????
        '''
        self.fea = fea_out

def get_feas_by_hook(model,hook_name):
    """
    ??????Conv2d??????feature??????????????????????????????module???????????????Conv2d??????hook?????????????????????module??????
    ???????????????????????????????????????Conv2d???????????????hook_fun?????????????????????feature.
    ????????????????????????????????????Conv2d?????????????????????hook_feas?????????????????????Conv2d??????feature
    """
    fea_hooks = []
    for n, m in model.named_modules():
        if n in hook_name:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks

def modified_model_inference(model,img,model_name):
    if model_name=='DBnet':
        score_text = model(img)
    elif model_name=='CRAFT':
        score_text = get_CRAFT_pred(model, img=img,square_size=1280, is_eval=False,)
    return score_text

def single_grad_inference(model,img ,feaName,model_name):

    if feaName and len(feaName) != 0:
        fea_hooks = get_feas_by_hook(model, hook_name=feaName)
    pp_result = modified_model_inference(model, img,model_name)
    if feaName and len(feaName)!=0:
        return pp_result,fea_hooks
    return pp_result