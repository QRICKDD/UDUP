import torch
from PIL import Image
from mmocr.utils.ocr import MMOCR
from torchvision import transforms

to_tensor = transforms.ToTensor()


if __name__ == "__main__":

    # 导入模型到内存
    ocr = MMOCR(det='DB_r18', recog=None)
    img = Image.open(r'F:\OCR-TASK\OCR_Dataset\Other_back\0.png')
    results,feamap = ocr.readtext(to_tensor(img),
                                  output='./', export='./',
                                  feaName=["neck.lateral_convs.0",
                                           "neck.lateral_convs.1",
                                           "neck.lateral_convs.2",
                                           "neck.lateral_convs.3",
                                           "bbox_head.binarize.7"])

    print(feamap[0].fea.shape)
    print(feamap[1].fea.shape)
    print(feamap[2].fea.shape)
    print(feamap[3].fea.shape)
    print(torch.max(feamap[4].fea))

