#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_msgs.msg import Int32MultiArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from model import FaceRCNN
import torch


class pytorch_detector:

    def __init__(self):

        self.img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_cb)

        self.img_pub = rospy.Publisher('/person_img', Image, queue_size=1)

        self.bridge = CvBridge()

        self.model = FaceRCNN()


    
    def img_cb(self, img_msg):

        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        self.detect(img)

    def detect_and_draw(self):

        try:
            boxes, labels, boxed_img = self.model.person_boxes(self.img)
            print(boxes)
            print(labels)
            boxed_img_msg = self.bridge.cv2_to_imgmsg(boxed_img, 'bgr8')

            self.img_pub.publish(boxed_img_msg)
        except:
            pass

    def detect(self, img):

        try:
            first_img = img
            boxes, labels, scores = self.model.simple_detection(first_img)
            for i, box in enumerate(boxes):
                if scores[i] > 0.4:
                    if labels[i] == 1:
                        cv2.rectangle(first_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 4)
                    else:
                        cv2.rectangle(first_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 4)
            imgmsg = self.bridge.cv2_to_imgmsg(first_img, 'bgr8')
            self.img_pub.publish(imgmsg)
            print("Published image")
        except Exception as e:
            print(e)


# def resize(img):
#     w = img.shape[0]
#     h = img.shape[1]


if __name__ == '__main__':

    rospy.init_node('detector')

    detector = pytorch_detector()

    rospy.spin()
    

