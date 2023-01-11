import cv2
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np
from Tools.ImageIO import img_tensortocv2

def img_show3_t(img):
    img=img_tensortocv2(img)
    plt.imshow(img)
    plt.show()

def img_show3(img):
    plt.matshow(img)
    plt.show()

def img_show1_t(img):
    plt.imshow(img.detach().cpu().squeeze())
    plt.show()

def img_show1(img):
    plt.matshow(img, cmap=plt.cm.gray)
    plt.show()

def img_grad_show(img: torch.Tensor) -> None:
    assert img.requires_grad == True
    ygf = img.grad_fn
    print('')
    print('***********cyclic print grads**************')
    while ygf != ():
        print(ygf)
        try:
            if ygf.next_functions[0][0]==None and len(ygf.next_functions)>1:
                ygf = ygf.next_functions[1][0]
            else:
                ygf = ygf.next_functions[0][0]

        except:
            break


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

def CRAFT_draw_box(img,boxes):
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    return img

def Draw_box(img,boxes,save_path,model_name):
    assert type(img) == np.ndarray
    if model_name=='CRAFT':
        img = CRAFT_draw_box(img, boxes)
    else:
        img = draw_bbox(img, boxes)
    cv2.imwrite(save_path, img)
