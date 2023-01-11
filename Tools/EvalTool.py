import os
import numpy as np
import math
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        matchedSum = 0

        numGlobalCareGt = 0
        numGlobalCareDet = 0


        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        detMatchedNums = []


        # evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]
            # transcription = gt[n]['text']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)

        for n in range(len(pred)):
            points = pred[n]
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)


        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            detMatchedNums.append(detNum)

        numGtCare = len(gtPols)
        numDetCare = len(detPols)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
            precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum)/numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum)/numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
            methodRecall * methodPrecision / (methodRecall + methodPrecision)

        # methodMetrics = {'precision': methodPrecision,
        #                  'recall': methodRecall, 'hmean': methodHmean}

        return methodPrecision,methodRecall,methodHmean

def test_Detectionevaluator():
    evaluator = DetectionIoUEvaluator()
    gts = [
        [[[0, 0], [1, 0], [1, 1], [0, 1]],
         [[2, 2], [3, 2], [3, 3], [2, 3]],]
    ]
    preds = [
        [[[0.1, 0.1], [1, 0], [1, 1], [0, 1]], ]
    ]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    P,R,F = evaluator.combine_results(results)
    print(P,R,F)


def read_txt(txt_path):
    with open(txt_path,'r') as f:
        reader_lines=f.readlines()
    res=[]
    for line in reader_lines:
        lines = line.strip().split(',')
        lines=[int(item) for item in lines[:8]]
        poly = np.array(list(map(float, lines))).reshape((-1, 2)).tolist()
        res.append(poly)
    return res

# def read_dir_txt(dir_path):
#     all_files=os.listdir(dir_path)
#     all_files=[os.path.join(dir_path,item) for item in dir_path]
#     pass


import cv2

def get_mask(file_path,image_size):
    points = read_txt(file_path)
    #points=np.array(points)
    im = np.zeros(image_size)
    cv2.fillPoly(im, np.array(points,np.int32), 1)
    #img_show1(im)
    return im

def test_get_mask():
    from Tools.Showtool import img_show1
    test_file = '../AllConfig/all_data/test_gtx/test.txt'
    image_size=(1000,1000)
    im=get_mask(test_file,image_size)
    img_show1(im)
    print("OK")


def test_read_txt():
    test_file='../AllConfig/all_data/test_gtx/test.txt'
    res=read_txt(test_file)
    print(res)




