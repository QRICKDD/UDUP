def test_move_clip_image():
    import os
    from PIL import Image

    # set the path to the folder containing the images
    folder_path = r"C:\Users\djc\Documents\GitHub\udup_djc\AllData\real-world"

    # set the desired dimensions
    width = 1300
    height = 1100

    # set the path to save the processed images
    save_path = r"C:\Users\djc\Documents\GitHub\udup_djc\real_world"

    # loop through each file in the folder
    for filename in os.listdir(folder_path):
        # check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # open the image
            image = Image.open(os.path.join(folder_path, filename))
            # get the current dimensions
            current_width, current_height = image.size
            # check if the image needs to be resized
            if current_width < width or current_height < height:
                # calculate the new dimensions while maintaining aspect ratio
                ratio = min(width / current_width, height / current_height)
                new_width = int(current_width * ratio)
                new_height = int(current_height * ratio)
                # resize the image
                image = image.resize((new_width, new_height))
            # crop the image from the left side to the desired dimensions
            left = 0
            upper = 0
            right = width
            lower = height
            image = image.crop((left, upper, right, lower))
            # save the processed image
            image.save(os.path.join(save_path, filename))

def test_change_background():
    import cv2
    import numpy as np

    # 读取图片
    img = cv2.imread(r'C:\Users\djc\Documents\GitHub\udup_djc\AllData\real-world\english.png')

    # 统计三通道之和
    sum_channels = np.sum(img, axis=2)

    # 统计出现次数最多的和
    most_common_sum = np.argmax(np.bincount(sum_channels.flatten()))

    # 将对应位置的像素改成白色
    white_pixels = np.where(sum_channels == most_common_sum)
    img[white_pixels] = [255, 255, 255]

    # 保存图片
    cv2.imwrite(r'C:\Users\djc\Documents\GitHub\udup_djc\AllData\real-world\english.png', img)

