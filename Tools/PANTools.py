import string
import argparse
import json
import os
import os.path as osp
import sys
from AllConfig.GConfig import abspath

import torch
from mmcv import Config
from model_PAN.models import build_model
from model_PAN.models.utils import fuse_module
from model_PAN.mydata import preprocess_pan_image
def load_PANPlusmodel():
    args = argparse.Namespace(checkpoint=os.path.join(abspath,"AllConfig/all_model/panpp_r18_joint_train.pth.tar"),
        config=os.path.join(abspath,'AllConfig/all_model/pan_pp_r18_ic15_736_joint_train.py'),
        debug=False, report_speed=False, vis=False)
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
    cfg.update(dict(vis=args.vis))
    cfg.update(dict(debug=args.debug))
    cfg.data.test.update(dict(debug=args.debug))
    #print(json.dumps(cfg._cfg_dict, indent=4))

    EOS = 'EOS'
    PADDING = 'PAD'
    UNKNOWN = 'UNK'
    voc=list(string.digits + string.ascii_lowercase)
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)
    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    if hasattr(cfg.model, 'recognition_head'):
        cfg.model.recognition_head.update(dict(voc=voc,char2id=char2id,id2char=id2char,))
    model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(
                args.checkpoint))

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model.eval()
    return model,cfg


def pan_preprocess_image(img_tensor):
    data=preprocess_pan_image(img_tensor)
    return data
