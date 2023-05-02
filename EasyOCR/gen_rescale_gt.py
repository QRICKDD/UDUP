import easyocr
import torchvision.transforms as transforms
import cv2
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
reader = easyocr.Reader(['en'], gpu=True,model_storage_directory=r'../AllConfig/all_model')
def get_result(result):
    points=[]
    for item in result:
        points.append(item[0])
    return points

def write_res(file_path,res):
    with open(file_path,'w') as f:
        for item in res:#这是一组8个[[a,b],[c,d]...]
            ite=[str(int(temp[0]))+","+str(int(temp[1])) for temp in item]
            f.writelines(",".join(ite)+'\n')
    return


dir_path=r"../AllData/test_resize"
import os
all_images=os.listdir(dir_path)
all_abs_images=[os.path.join(dir_path,item) for item in all_images]


import cv2
gt_dir_path=r"../AllData/test_resize_gt"
temp_path=r"../AllData/temp"

resize_scales = [item / 10 for item in range(6, 21, 1)]  # 0.6 0.7 0.8 ... 2.0


from Tools.ImageIO import *
for imgname,path_item in zip(all_images,all_abs_images):
    for item in resize_scales:
        str_s = str(int(item * 100))
        save_rr = os.path.join(gt_dir_path, str_s)
        if os.path.exists(save_rr) == False:
            os.makedirs(save_rr)
        temp_save_path = os.path.join(temp_path,imgname)#保存resize图的暂时路径
        img_torch=img_read(path_item)
        h,w=img_torch.shape[2:]
        resize_cvimg=img_tensortocv2(transforms.Resize([int(h*item),int(w*item)])(img_torch))
        cv2.imwrite(temp_save_path,resize_cvimg)
        #time.sleep(1)
        result = reader.readtext(temp_save_path)
        res=get_result(result)#获取列表
        #目标txt路径=目标路径+图片名称_gt.txt
        target_txt=os.path.join(save_rr,imgname.split(".")[0]+"_gt.txt")
        write_res(target_txt,res)




