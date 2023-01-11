import torch
DBnet_model_path=r"../AllConfig/all_model/DBNet_shufflenetv2.pth"
DBnet_config_json_path=r'../AllConfig/config_DBnet/config.json'
test_img_path= r'../AllData/all_data/test_image/0.png'
CRAFT_model_path='../AllConfig/all_model/craft_ic15_20k.pth'

#配置DBnet网络和数据存放的位置：
import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"
DB_device=torch.device("cuda:0")
CRAFT_device=torch.device("cuda:0")
test_device=torch.device("cuda:0")
cpu_device=torch.device("cpu")
#data.to(device)
