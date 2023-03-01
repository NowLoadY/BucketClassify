"""
NowLoadY
llzhandsome2@gmail.com
将JPEGImages文件夹内所有照片转换为灰度图像，保存为png格式，保存到output文件夹。
"""
import cv2
import os
import time

x = 0
for root, dirs, files in os.walk('JPEGImages'):
    for d in dirs:
        print(d)  # 打印子资料夹的个数
    for file in files:
        time.sleep(0.001)
        img_path = root + '/' + file
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 保存
        if not os.path.exists('output'):
            os.makedirs('output')
        print(file)
        img_saving_path = 'output/" + file
        cv2.imwrite(img_saving_path, img)
