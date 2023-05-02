import torch
import cv2
from torchvision import transforms
def img_read(image_path) -> torch.Tensor:
    transform = transforms.ToTensor()
    im = cv2.imread(image_path, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # img_show3(im)
    img = transform(im)
    img = img.unsqueeze_(0)
    return img

def repeat_4D(patch: torch.Tensor,h_real, w_real) -> torch.Tensor:
    assert (len(patch.shape) == 4 and patch.shape[0] == 1)
    #assert patch.requires_grad == True
    patch_h,patch_w=patch.shape[2:]
    h_num=h_real//patch_h+1
    w_num = w_real // patch_w+1
    patch = patch.repeat(1, 1, h_num, w_num)
    patch = patch[:, :, :h_real, :w_real]
    return patch

def extract_background(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_sum = torch.sum(img_tensor, dim=1)
    mask = (img_sum == 3)
    mask = mask + 0
    return mask.unsqueeze_(0)

def img_tensortocv2(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_tensor = img_tensor.clone().detach().cpu()
    img_tensor = img_tensor.squeeze()
    img_tensor = img_tensor.mul_(255).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img_cv = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)
    return img_cv