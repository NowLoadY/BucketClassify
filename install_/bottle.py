#!/usr/bin/python
# -*- coding:UTF-8 -*-
"""
Project: Bottle classification
Author: NowLoadY, Seng Mao, Bo Leng
"""
from ctypes import *
import random
import cv2
import time
import darknet
from threading import Thread
from queue import Queue
import tools
import RPi.GPIO as GPIO
import time
import sys


def video_capture():
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)

    cap.release()

    

def inference():
    global Controled
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh, nms=nms)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)

        if print_Fps:
            print("FPS: {}".format(fps))

        if len(detections) <= maxObjectNum:
            for label, confidence, bbox in detections:
                x,y,w,h=bbox
                if label=="3":
                        controlGPIO("class1")
                        print("class1")
                elif left_bounce < x < right_bounce:
                    thresh_height = left_thresh_height+(x-left_bounce)*liner_k
                    if y<thresh_height:
                        height_status="HIGH"
                    else:
                        height_status="LOW"
                    if label=="1":
                        controlGPIO("class3")
                    elif label=="2":
                        if height_status=="HIGH":
                            controlGPIO("class2")
                        elif height_status=="LOW":
                            controlGPIO("class1")
                else:
                    Controled = False
        
        if print_Detections:
            darknet.print_detections(detections)
        darknet.free_image(darknet_image)
    cap.release()


def drawing():
    random.seed(3)  # deterministic bbox colors
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = tools.convert2original(frame, bbox, darknet_width, darknet_height)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
                if show_img:# 绘制目标框中心点和阈值高度参考点
                    x,y,w,h = bbox
                    thresh_height = (x-left_bounce)*liner_k+left_thresh_height
                    frame = cv2.circle(frame, (int(x),int(thresh_height)), 10, (0,0,255),-1)
                    frame = cv2.circle(frame, (int(x),int(y)), 10, (255,0,0),-1)
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if show_img:
                # 画辅助线
                image = draw_ui(image)
                # 显示图像
                cv2.imshow('Inference', image)
            if cv2.waitKey(fps if fps!=0 else 10) == 27:
                break
    cap.release()
    if show_img:
        cv2.destroyAllWindows()
    sys.exit(0)


def controlGPIO(bottle_class):
    global Controled
    if not Controled:
        if print_Detections:
            print("output:{}".format(bottle_class))
        if bottle_class == 'class1':
            GPIO.output(pin_1, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(pin_1, GPIO.LOW)
        elif bottle_class == 'class2':
            GPIO.output(pin_2, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(pin_2, GPIO.LOW)
        elif bottle_class == 'class3':
            GPIO.output(pin_3, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(pin_3, GPIO.LOW)
        else:
            GPIO.output(pin_1, GPIO.LOW)
            GPIO.output(pin_2, GPIO.LOW)
            GPIO.output(pin_3, GPIO.LOW)
        Controled = True


def draw_ui(image):
    image = cv2.line(image,(int(left_bounce),0), (int(left_bounce),darknet_height), (255, 0 , 0),1)
    image = cv2.line(image,(int(right_bounce),0), (int(right_bounce),darknet_height), (255, 0 , 0),1)
    image = cv2.line(image,(int(left_bounce),int(left_thresh_height)), (int(right_bounce),int(right_thresh_height)), (0, 255 , 0),3)
    return image



if __name__ == '__main__':
    # ---------初始化文件路径--------- #
    config_file = "./cfg/yolov4-tiny.cfg"
    data_file = "./cfg/bottle.data"
    weights = "bottleV4.weights"
    # ---------设置检测参数--------- #
    thresh = 0.85
    nms = 0.01
    maxObjectNum = 1
    CamPath = 0
    # ---------设置可视化--------- #
    print_Fps = False
    print_Detections = False
    show_img = True
    # ---------设置GPIO口--------- #
    pin_1 = 32  # class1
    pin_2 = 31  # class2
    pin_3 = 29  # class3
    #high_pin = 33
    #but_pin = 18
    # ---------初始化GPIO--------- #
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup([pin_1, pin_2, pin_3], GPIO.OUT)  # LED pins set as output
    GPIO.output(pin_1, GPIO.LOW)
    GPIO.output(pin_2, GPIO.LOW)
    GPIO.output(pin_3, GPIO.LOW)
    # ---------状态变量--------- #
    Controled = False
    height_status = "none"
    # ---------初始化队列--------- #
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    # ---------加载模型--------- #
    network, class_names, class_colors = darknet.load_network(config_file, data_file, weights, batch_size=1)
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    # ---------打开摄像头--------- #
    cap = cv2.VideoCapture(tools.str2int(CamPath))
    #cap.set(3,360)
    #cap.set(4,640)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #----------阈值----------#
    left_bounce=darknet_width*0.4
    right_bounce=darknet_width*0.6
    liner_k = 0
    left_thresh_height = 330
    #----------参数计算----------#
    bounce_width = right_bounce-left_bounce
    right_thresh_height= left_thresh_height + bounce_width*liner_k
    # ---------开启多线程--------- #
    Thread(target=video_capture).start()
    Thread(target=inference).start()
    Thread(target=drawing).start()
