import torch
import sys
import os
abspath=os.path.abspath(r'K:\udup_djc')
test_img_path= os.path.join(abspath,'AllData/test/020.png')
CRAFT_model_path=os.path.join(abspath,'AllConfig/all_model/craft_ic15_20k.pth')


#配置DBnet网络和数据存放的位置：
import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"
DB_device=torch.device("cuda:0")
CRAFT_device=torch.device("cuda:0")
test_device=torch.device("cuda:0")
cpu_device=torch.device("cpu")
#data.to(device)
