import os
import os.path as osp
import cv2
import numpy as np
from model_PAN.Ax import *

class Visualizer:
    def __init__(self, vis_path):
        self.vis_path = vis_path
        if not osp.exists(vis_path):
            os.makedirs(vis_path)

    def process(self, img_metas, outputs):
        img_path = img_metas['img_path'][0]
        img_name = img_metas['img_name'][0]
        img_name=os.path.basename(img_path).split('.')[0]
        bboxes = outputs['bboxes']
        if 'words' in outputs:
            words = outputs['words']
        else:
            words = [None] * len(bboxes)

        img = cv2.imread(img_path)
        for bbox, word in zip(bboxes, words):
            cv2.drawContours(img, [bbox.reshape(-1, 2)], -1, (0, 255, 0), 2)
            if word is not None:
                pos = np.min(bbox.reshape(-1, 2), axis=0)
                cv2.putText(img, word, (pos[0], pos[1]),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imwrite(osp.join(self.vis_path, '%s.jpg' % img_name), img)

    def process2(self, img_metas, outputs,adv_path):
        img_path = img_metas['img_path'][0]
        img_name = img_metas['img_name'][0]
        img_name=os.path.basename(img_path).split('.')[0]
        bboxes = outputs['bboxes']
        if 'words' in outputs:
            words = outputs['words']
        else:
            words = [None] * len(bboxes)

        #img = cv2.imread(img_path)
        img = img_read(img_path)
        # h, w = img.shape[2:]
        # adv_patch = torch.load(adv_path)
        # UAU = repeat_4D(adv_patch.clone().detach(), h, w)
        # mask_t = extract_background(img)
        # merge_image = img * (1 - mask_t) + mask_t * UAU
        #img=img_tensortocv2(merge_image)
        img = img_tensortocv2(img)

        for bbox, word in zip(bboxes, words):
            cv2.drawContours(img, [bbox.reshape(-1, 2)], -1, (0, 255, 0), 2)
            if word is not None:
                pos = np.min(bbox.reshape(-1, 2), axis=0)
                cv2.putText(img, word, (pos[0], pos[1]),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imwrite(osp.join(self.vis_path, '%s.jpg' % img_name), img)
