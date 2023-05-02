from Tools.CRAFTTools import *

img_test_1 = r"F:\udup\AllData\test\019.png"
img_test_2 = r"F:\udup\AllData\test\016.png"
if __name__ == '__main__':
    from Tools.ImageIO import *
    from AllConfig.GConfig import test_img_path,CRAFT_device
    #加载模型
    CRAFTnet=load_CRAFTmodel()
    #加载图片
    # img_path=test_img_path
    img_path=img_test_2
    img=img_read(img_path)
    img=img.to('cuda:0')

    #预测结果
    score_text,score_link,target_ratio = get_CRAFT_pred(CRAFTnet,img=img,square_size=1280,is_eval=True)
    boxes=get_CRAFT_box(score_text,score_link,target_ratio,
                        text_threshold=0.7,link_threshold=0.4,low_text=0.4)
    # 保存结果到cv图片
    img = cv2.imread(img_path)
    cv2.imwrite(r"..\test_save\craft_orgin.png",img)
    CRAFT_draw_box(img,boxes=boxes,save_path=r"..\test_save\craft_boxes.png")
