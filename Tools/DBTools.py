from model_DBnet.models.model import Model as FPNModel
from AllConfig import GConfig
import os
import torch
import numpy as np
import pyclipper
import cv2

def load_DBmodel():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #加载模型
    DBnet_path=GConfig.DBnet_model_path
    checkpoint = torch.load(DBnet_path)
    config = {
            "backbone": "shufflenetv2",
            "fpem_repeat": 2,
            "pretrained": False,
            "segmentation_head": "FPN"
    }
    DBnet = FPNModel(config)
    DBnet.load_state_dict(checkpoint['state_dict'])
    DBnet=DBnet.to('cuda:0')
    DBnet.eval()
    return DBnet

def get_DB_dilateds_boxes(preds:torch.Tensor,h,w,min_area=100):
    scale = (preds.shape[2] / w, preds.shape[1] / h)
    prob_map, thres_map = preds[0], preds[1]
    ## Step 1: Use threshold to get the binary map
    thr = 0.2
    out = (prob_map > thr).float() * 255
    out = out.data.cpu().numpy().astype(np.uint8)
    ## Step 2: Connected components findContours
    contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = [(i / scale).astype(np.int) for i in contours if len(i) >= 4]
    contours = [(i/scale).astype(np.int) for i in contours if len(i) >= 4]
    # Step 3: Dilate the shrunk region (not necessary)
    ratio_prime = 1.5
    dilated_polys = []
    for poly in contours:
        poly = poly[:, 0, :]
        D_prime = cv2.contourArea(poly) * ratio_prime / cv2.arcLength(poly, True)  # formula(10) in the thesis
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated_poly = np.array(pco.Execute(D_prime))
        if dilated_poly.size == 0 or dilated_poly.dtype != np.int or len(dilated_poly) != 1:
            continue
        dilated_polys.append(dilated_poly)
    boxes_list = []
    for cnt in dilated_polys:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = (cv2.boxPoints(rect)).astype(np.int)
        boxes_list.append(box)
    boxes_list = np.array(boxes_list)
    return dilated_polys,boxes_list

def draw_bbox(img, result, color=(128, 240, 128), thickness=3):
    assert type(img) == np.ndarray
    if isinstance(img, str):
        img = cv2.imread(img)
    for point in result:
        point = point.astype(int)
        cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)
    return img

def DB_draw_dilated(img,dilateds,save_path):
    assert type(img)==np.ndarray
    if isinstance(img, str):
        img = cv2.imread(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgc=img.copy()
    cv2.drawContours(imgc, dilateds, -1, (22, 222, 22), 2, cv2.LINE_AA)
    cv2.imwrite(save_path,imgc)

def DB_draw_box(img,boxes,save_path):
    assert type(img) == np.ndarray
    img = draw_bbox(img, boxes)
    cv2.imwrite(save_path, img)


