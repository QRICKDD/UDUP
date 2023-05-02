import torch
from torch import nn
from model_DBnet.models.modules.shufflenetv2 import shufflenet_v2_x1_0
from model_DBnet.models.modules.segmentation_head import FPN,FPEM_FFM

backbone_dict = {
                 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]}
                 }
"""
FPN 金字塔   FPEM 特征增强  FFM融合特征   PANnet结构，是PSEnet的优化版本

相关信息参考  https://blog.csdn.net/c991262331/article/details/109320811
"""
segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM}


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_head_out = self.segmentation_head(backbone_out)
        y = segmentation_head_out
        return y