from Tools.DBTools import *

img_test_1 = r"F:\OCR-TASK\Wsf\data\test\002.png"
img_test_3 = r"F:\OCR-TASK\Wsf\data\test\003.png"
img_test_2 = r"F:\OCR-TASK\Wsf\data\test\006.png"
if __name__ == '__main__':
    from Tools.Imagebasetool import *
    from AllConfig.GConfig import test_img_path,test_device
    #加载模型
    DBnet=load_DBmodel(device=test_device,feamap=True)
    #加载图片
    # img_path=test_img_path
    img_path = img_test_3
    img=img_read(img_path)
    img=img.cuda()
    # preds = DBnet(img)[0]
    preds,feamap = DBnet(img)




    #保存结果到cv图片
    img = cv2.imread(img_path)
    # 根据结果计算图
    h,w=img.shape[:-1]
    dilates, boxes = get_DB_dilateds_boxes(preds,h,w, min_area=100)
    cv2.imwrite(r"..\result_save\test_save\orgin.jpg",img)
    DB_draw_dilated(img,dilateds=dilates,save_path=r"..\result_save\test_save\dilated.jpg")
    DB_draw_box(img,boxes=boxes,save_path=r"..\result_save\test_save\boxes.jpg")

    #测试图片


