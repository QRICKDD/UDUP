import torch
from Tools.ImageIO import img_tensortocv2,img_read
from Tools.Baseimagetool import repeat_4D
from UDUP.Auxiliary import extract_background

iter_dict={10:[18,21,24,26,29],
           20:[22,25,28,31,34],
           30:[21,25,29,32,35],
           50:[22,27,30,33,37],
           80:[24,27,30,33,37],
           100:[23,27,30,33,38],
           150:[24,28,32,36,41],
           200:[27,30,34,39,45]}
xnames=[0.06,0.07,0.08,0.09,0.10]
for key,iters in iter_dict.items():
    for iter,xname in zip(iters,xnames):
        patch_path=r"F:\udup\result_save\size={}_step=3_eps=90_lambdaw=2.5\advtorch\advpatch_{}".\
            format(str(key),str(iter))
        img_path=r'F:\udup\AllData\test\019.png'
        save_path=r"F:\udup\temp-save"
        img=img_read(img_path)
        adv_patch=torch.load(patch_path)
        h,w=img.shape[2:]
        adv_patch_repeat=repeat_4D(adv_patch,h,w)
        mask=extract_background(img)
        merge_img=mask*adv_patch_repeat+(1-mask)*img
        import cv2
        import os
        cv2.imwrite(os.path.join(save_path,"{}-{}.png".format(str(key),str(xname))),img_tensortocv2(merge_img))
