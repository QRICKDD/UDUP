import warnings
import mmcv
from Tools.ImageIO import img_tensortocv2,img_read
warnings.filterwarnings('ignore')
from mmocr.utils.ocr import MMOCR
import torch
from Tools.ImageIO import *
#img=img_tensortocv2(img_read())
test_path='F:\\udup_djc\\AllData/test/020.png'
m1=mmcv.imread(test_path)
m2= img_tensortocv2(img_read(test_path))
ocr=MMOCR(det='PANet_IC15',recog=None)
preboxes=ocr.readtext2('F:\\udup_djc\\AllData/test/020.png')
print(preboxes)