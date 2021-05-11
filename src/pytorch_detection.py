#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from model import FaceRCNN
import torch


class pytorch_detector:

    def __init__(self):

        self.bridge = CvBridge()
        
        self.model = FaceRCNN(mobile_net=True)

        self.img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_cb)

        self.img_pub = rospy.Publisher('/person_img', Image, queue_size=1)

        self.person_pub = rospy.Publisher('/person_centroid', Int32MultiArray, queue_size=1)

        self.face_pub = rospy.Publisher('/face_detected', Bool, queue_size=1)



    def img_cb(self, img_msg):

        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        self.detect(img)

    def publish_data(self, boxes, labels, scores):
        max_score = 0
        cX = None
        cY = None
        face_detected = False
        for i, box in enumerate(boxes):
            if scores[i] > max_score and labels[i] == 1:
                cX = int(box[0] + ((box[2]-box[0])/2))
                cY = int(box[1] + ((box[3]-box[1])/2))
                max_score = scores[i]
            elif labels[i] == 2:
                face_detected = True
        if cX != None and cY != None:
            person_msg = Int32MultiArray(data=[cX, cY])
        else:
            person_msg = Int32MultiArray(data=[-1, -1])
        face_msg = Bool(data=face_detected)
        self.person_pub.publish(person_msg)
        self.face_pub.publish(face_msg)
        


    def detect(self, img):

        try:
            boxes, labels, scores = self.model.simple_detection(img)
            self.publish_data(boxes, labels, scores)
            for i, box in enumerate(boxes):
                if scores[i] > 0.4:
                    if labels[i] == 1:
                        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 4)
                    else:
                        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 4)
            imgmsg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            self.img_pub.publish(imgmsg)
            print("Published image")
        except Exception as e:
            print(e)



if __name__ == '__main__':

    rospy.init_node('detector')

    detector = pytorch_detector()

    rospy.spin()
    

