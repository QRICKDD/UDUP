import os

import cv2
import numpy as np
from Tools.ImageIO import *
from model_CRAFT.pred_single import *
from AllConfig.GConfig import abspath
import matplotlib.pyplot as plt
from torchvision import transforms
from UDUP.Auxiliary import *
from Tools.EvalTool import *
import warnings
warnings.filterwarnings("ignore")

CRAFTnet=load_CRAFTmodel()
transform = transforms.ToTensor()


def compress_image(img_matrix, quality):
    # 将图像矩阵转换为JPEG格式
    _, compressed_data = cv2.imencode('.jpg', img_matrix, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # 将压缩后的数据转换回图像矩阵
    compressed_img = cv2.imdecode(np.frombuffer(compressed_data, np.uint8), cv2.IMREAD_COLOR)

    return compressed_img

def read_txt(txt_path):
    with open(txt_path,'r') as f:
        reader_lines=f.readlines()
    res=[]
    for line in reader_lines:
        lines = line.strip().split(',')
        lines=[int(item) for item in lines[:8]]
        poly = np.array(list(map(float, lines))).reshape((-1, 2)).tolist()
        res.append(poly)
    return res

def evaluate_and_draw(adv_patch,image_root, gt_root,Q):
    evaluator=DetectionIoUEvaluator()
    global CRAFTnet
    CRAFTnet=CRAFTnet.eval()
    image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
    images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
    test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]
    results = []  # PRF


    for img, name, gt in zip(images, image_names, test_gts):
        print(name)
        h, w = img.shape[2:]

        UAU = repeat_4D(adv_patch.clone().detach().cpu(), h, w)
        mask_t = extract_background(img)
        img = img
        merge_image = img * (1 - mask_t) + mask_t * UAU
        merge_image = merge_image
        jpeg_img=compress_image(img_tensortocv2(merge_image),Q)
        jpeg_img_tensor = transform(jpeg_img)
        merge_image = jpeg_img_tensor.unsqueeze(0)
        merge_image=merge_image.cuda()

        (score_text, score_link, target_ratio) = single_grad_inference(CRAFTnet, merge_image, [],
                                                                       'craft',is_eval=True)
        boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                              text_threshold=0.7, link_threshold=0.4, low_text=0.4)

        gt = read_txt(gt)
        temp = evaluator.evaluate_image(gt, boxes)
        results.append(temp)
        #
        # temp_save_path = os.path.join(save_path, name.split('\\')[-1])
        # adv_patch_path_better = os.path.join(abspath, "real_world", "100_mui_0.9")
        # adv_patch=torch.load(adv_patch_path_better)
        # UAU = repeat_4D(adv_patch.clone().detach().cpu(), h, w)
        # mask_t = extract_background(img)
        # img = img
        # merge_image = img * (1 - mask_t) + mask_t * UAU
        # merge_image = merge_image
        # jpeg_img = compress_image(img_tensortocv2(merge_image), Q)
        # jpeg_img_tensor = transform(jpeg_img)
        # merge_image = jpeg_img_tensor.unsqueeze(0)
        # merge_image = merge_image

        #Draw_box(img_tensortocv2(merge_image), boxes, temp_save_path, model_name='craft')
        del(merge_image,mask_t,UAU,jpeg_img_tensor)
        torch.cuda.empty_cache()
    P, R, F = evaluator.combine_results(results)
    return P, R, F

import os

adv_patch_path=[
        os.path.join(abspath,"result_save_0.06","size=10_step=3_eps=120_lambdaw=0.001","advtorch","advpatch_24"),
        os.path.join(abspath,"result_save_0.06","size=20_step=3_eps=120_lambdaw=0.1","advtorch","advpatch_27"),
        os.path.join(abspath,"result_save_0.06","size=30_step=3_eps=120_lambdaw=0.1","advtorch","advpatch_29"),
        os.path.join(abspath,"result_save_0.06","size=50_step=3_eps=120_lambdaw=0.001","advtorch","advpatch_29"),
        os.path.join(abspath,"result_save_0.06","size=100_step=3_eps=120_lambdaw=0.1","advtorch","advpatch_31"),
        os.path.join(abspath,"result_save_0.06","size=150_step=3_eps=120_lambdaw=0.1","advtorch","advpatch_33"),
        os.path.join(abspath,"result_save_0.06","size=200_step=3_eps=120_lambdaw=0.01","advtorch","advpatch_35")
    ]


adv_patch_path=os.path.join(abspath,"result_save_0.06","size=200_step=3_eps=120_lambdaw=0.01","advtorch","advpatch_35")
gt_path=os.path.join(abspath,"AllData/test_craft_gt")
test_path=os.path.join(abspath,"AllData/test")


image_file_names = os.listdir(test_path)
image_names = [os.path.join(test_path, name) for name in os.listdir(test_path)]
Qs=[50,60,70,80,90,100]
for Q in Qs:
    adv_patch=torch.load(adv_patch_path)
    # plt.imshow(jpeg_img)
    # plt.show()
    # cv2.imwrite("test.jpg", jpeg_img)
    # save_path=os.path.join(abspath,"jpeg_test","100_mui_0.9",str(Q))
    # if os.path.exists(save_path)==False:
    #     os.makedirs(save_path)
    P,R,F=evaluate_and_draw(adv_patch=adv_patch,image_root=test_path,gt_root=gt_path,Q=Q)
    print("Q:{},P:{},R:{}".format(Q,P,R))



