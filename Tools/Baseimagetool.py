import numpy as np
import torch
import random
from torchvision import transforms
import torch.nn as nn
from Tools.Showtool import *

#黑色是文字区域，白色是非文字区  提取背景，需要白色区域为1
#白色是文字区域，黑色是非文字区  提取文字区，需要白色区域为1
def extract_background(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_sum = torch.sum(img_tensor, dim=1)
    mask = (img_sum == 3)
    mask = mask + 0
    return mask.unsqueeze_(0)

def random_image_resize(image: torch.Tensor, low=0.25, high=3):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    #assert image.requires_grad == True
    scale = random.random()
    shape = image.shape
    h, w = shape[-2], shape[-1]
    h, w = int(h * (scale * (high - low) + low)), int(w * (scale * (high - low) + low))
    image = transforms.Resize([h, w])(image)
    return image

def repeat_4D(patch: torch.Tensor,h_real, w_real) -> torch.Tensor:
    assert (len(patch.shape) == 4 and patch.shape[0] == 1)
    #assert patch.requires_grad == True
    patch_h,patch_w=patch.shape[2:]
    h_num=h_real//patch_h+1
    w_num = w_real // patch_w+1
    patch = patch.repeat(1, 1, h_num, w_num)
    patch = patch[:, :, :h_real, :w_real]
    return patch

def random_noise(image: torch.Tensor,noise_low,noise_high):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    device=image.device
    temp_image=image.clone().detach().cpu().numpy()
    noise=np.random.uniform(low=noise_low,high=noise_high,size=temp_image.shape)
    noise=torch.from_numpy(noise)
    noise=noise.float()
    noise=noise.to(device)

    image=torch.clamp(image+noise,min=0,max=1)
    return image


#CRAFT的操作
def normlize_MeanVariance(image:torch.Tensor,device):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    mean = torch.Tensor([[[[0.485]],[[0.456]],[[ 0.406]]]])
    mean=mean.to(device)
    variance = torch.Tensor([[[[0.229]], [[0.224]], [[0.225]]]])
    variance=variance.to(device)
    image=(image-mean)/variance
    return image

#CRAFT的操作
def resize_aspect_ratio(image:torch.Tensor,square_size,mag_ratio=1.5):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    h,w=image.shape[2:]
    target_size = mag_ratio * max(h, w)
    if target_size>square_size:
        target_size=square_size
    ratio=target_size/max(h,w)
    target_h,target_w=int(h*ratio),int(w*ratio)
    image=transforms.Resize([target_h,target_w])(image)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = torch.zeros([1,3,target_h32, target_w32]).to('cuda:0')
    resized[:,:,0:target_h,0:target_w]=image

    #target_h, target_w = target_h32, target_w32
    #size_heatmap = (int(target_w / 2), int(target_h / 2))
    #return size_heatmap
    return resized,ratio






"""
============================================TEST================================================================
============================================TEST================================================================
"""


def test_repeat_4D():
    x = torch.Tensor([[[[0.4942, 0.1321],
                          [0.3797, 0.3320]]]])
    x.requires_grad = True
    img_h, img_w = 5, 5
    y = repeat_4D(x, img_h, img_w)

    referance=torch.Tensor([[[[0.4942, 0.1321,0.4942, 0.1321,0.4942],
                              [0.3797, 0.3320,0.3797, 0.3320,0.3797],
                              [0.4942, 0.1321,0.4942, 0.1321,0.4942],
                              [0.3797, 0.3320,0.3797, 0.3320,0.3797],
                              [0.4942, 0.1321,0.4942, 0.1321,0.4942],]]])
    assert (y==referance).all()
    img_grad_show(y)

def test_random_resize():
    from Tools.Showtool import img_grad_show
    img = torch.randn(1, 3, 120, 100)
    img.requires_grad = True
    img = random_image_resize(img, low=0.1, high=3)
    img_grad_show(img)




def test_resize_aspect_ratio():
    from Tools.Showtool import img_grad_show
    img=torch.randn(1,3,5,5)
    img.requires_grad=True
    img=img.cuda()
    resize_image,ratio=resize_aspect_ratio(image=img,device=torch.device('cuda:0'),
                                           square_size=10,mag_ratio=1.5)
    print(resize_image.shape)
    print(resize_image[0,0,:10,0])
    img_grad_show(resize_image)
