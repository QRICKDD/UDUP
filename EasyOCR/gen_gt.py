import easyocr
reader = easyocr.Reader(['en'], gpu=True,model_storage_directory=r'F:\OCR-TASK\OCR__advprotect\AllConfig\all_model')
# img_test_1 = r"../AllData/test/002.png"
# img_test_3 = r"../AllData/test/003.png"
# img_test_2 = r"../AllData/test/006.png"


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


dir_path=r"../AllData/test"
import os
all_images=os.listdir(dir_path)
all_abs_images=[os.path.join(dir_path,item) for item in all_images]


import cv2
gt_dir_path=r"../AllData/test_gt"
for imgname,item in zip(all_images,all_abs_images):
    img=cv2.imread(item)
    [h,w]=img.shape[1:]
    result = reader.readtext(item)
    res=get_result(result)#获取列表

    #目标txt路径=目标路径+图片名称_gt.txt
    target_txt=os.path.join(gt_dir_path,imgname.split(".")[0]+"_gt.txt")
    write_res(target_txt,res)

# img=cv2.imread(img_test_1)
# # [h,w]=img.shape[1:]
# result = reader.readtext(img_test_1)
# res=get_result(result)
# print(res)



