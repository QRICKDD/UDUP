import os
import torch
import torchvision.models as models
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import collections.abc as container_abcs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='', help='Input directory with images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--image_resize", type=int, default=330, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--amplification", type=float, default=10.0, help="To amplifythe step size.")
parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")

opt = parser.parse_args()

transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2

def project_noise(x):
    stack_kern, padding_size = project_kern(3)
    x = F.conv2d(x.cuda(), stack_kern.cuda(), padding = (padding_size, padding_size), groups=3)
    return x



def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def graph(x, gt, x_min, x_max):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    alpha_beta = alpha * opt.amplification
    gamma = alpha_beta

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.inception_v3(pretrained=True).eval().cuda())
    x.requires_grad = True
    amplification = 0.0
    for i in range(num_iter):
        zero_gradients(x)
        output_v3 = model(x)
        loss = F.cross_entropy(output_v3, gt)
        loss.backward()
        noise = x.grad.data

        # MI-FGSM
        # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
        # noise = momentum * grad + noise
        # grad = noise

        amplification += alpha_beta * torch.sign(noise)
        cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
        projection = gamma * torch.sign(project_noise(cut_noise))
        amplification += projection

        # x = x + alpha * torch.sign(noise)
        x = x + alpha_beta * torch.sign(noise) + projection
        x = clip_by_tensor(x, x_min, x_max)
        x = V(x, requires_grad = True)

    return x.detach()


def input_diversity(input_tensor):
    rnd = torch.randint(opt.image_width, opt.image_resize, ())
    rescaled = F.interpolate(input_tensor, size = [rnd, rnd], mode = 'bilinear', align_corners=True)
    h_rem = opt.image_resize - rnd
    w_rem = opt.image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [opt.image_resize, opt.image_resize])
    return padded if torch.rand(()) < opt.prob else input_tensor



class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]