
import json
import os
import os.path as osp
import sys

import torch

from model_PAN.utils import AverageMeter, Corrector, ResultFormat, Visualizer
from model_PAN.mydata import read_one_image
from Tools.PANTools import *
from Tools.ImageIO import *
def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para / 1e6))
    print('-' * 90)
def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))

import numpy as np

def test(model, cfg,dir_path,patch_path):
    model.eval()

    with_rec = hasattr(cfg.model, 'recognition_head')
    if with_rec:
        pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)

    if cfg.vis:
        vis = Visualizer(vis_path=osp.join('vis/', cfg.data.test.type))

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_post_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500))


    for fname in os.listdir(dir_path):
        data=read_one_image(os.path.join(dir_path,fname),patch_path=patch_path)
        data['imgs'] = data['imgs'].cuda()
        data['img_metas']['org_img_size']=np.array([data['img_metas']['org_img_size']])
        data['img_metas']['img_size'] = np.array([data['img_metas']['img_size']])
        data['img_metas']['img_path'] = [data['img_metas']['img_path']]
        data['img_metas']['img_name'] = [data['img_metas']['img_name']]
        data.update(dict(cfg=cfg))
        with torch.no_grad():
            outputs = model(**data)

        if with_rec:
            outputs = pp.process(data['img_metas'], outputs)

        # save result
        rf.write_result(data['img_metas'], outputs)

        # visualize

        vis.process2(data['img_metas'], outputs,patch_path)

    print('Done!')

def pan_pred_single(model,cfg,data):
    with_rec = hasattr(cfg.model, 'recognition_head')
    if with_rec:
        pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)

    if cfg.vis:
        vis = Visualizer(vis_path=osp.join('vis/', cfg.data.test.type))

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_post_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500))

    data['imgs'] = data['imgs'].cuda()
    data['img_metas']['org_img_size']=np.array([data['img_metas']['org_img_size']])
    data['img_metas']['img_size'] = np.array([data['img_metas']['img_size']])
    data.update(dict(cfg=cfg))
    with torch.no_grad():
        outputs = model(**data)

    # if with_rec:
    #     outputs = pp.process(data['img_metas'], outputs)

    # save result
    return outputs['bboxes']


def test_main():
    from AllConfig.GConfig import test_img_path,abspath

    model,cfg=load_PANPlusmodel()
    #model_structure(model)
    img=img_read(test_img_path)
    data=pan_preprocess_image(img)
    boxes=pan_pred_single(model,cfg,data)
    print(boxes)
    # test(model, img,cfg,dir_path="test2",
    #      patch_path=r"F:\udup\model_PAN\100advpatch_120")



