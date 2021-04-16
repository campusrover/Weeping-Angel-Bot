#!/usr/bin/env python

import numpy as np 
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError


class personDetector:

    def __init__(self):

        self.gray_img_sub = rospy.Subscriber('/gray_img', Image, self.gray_img_cb)

        self.color_img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.color_img_cb)

        self.person_pub = rospy.Publisher('/people', Int32MultiArray, queue_size=1)

        self.img_pub = rospy.Publisher('/person_img', Image, queue_size=1)

        self.person_arr = Int32MultiArray()

        self.bridge = CvBridge()
        
        self.hog = cv2.HOGDescriptor()

        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    def gray_img_cb(self, img):
        self.gray_img = self.bridge.imgmsg_to_cv2(img)

    def color_img_cb(self, color_img):
        self.color_img = self.bridge.imgmsg_to_cv2(color_img, 'bgr8')

    def detect_people(self):
        
        person_list = []
        people, _ = self.hog.detectMultiScale(self.gray_img, winStride=(8,8), scale=1)
        for (x, y, w, h) in people:
            mass = w*h
            person_list.append([x, mass])
            cv2.rectangle(self.color_img,(x,y),(x+w,y+h),(0,0,255),2)
        color_img_msg = self.bridge.cv2_to_imgmsg(self.color_img, 'bgr8')
        self.img_pub.publish(color_img_msg)
        self.person_arr.data = person_list
        self.person_pub.publish(self.person_arr)

if __name__ == "__main__":
    
    rospy.init_node('person_detector')
    person_detector = personDetector()
    rospy.sleep(1)
    while not rospy.is_shutdown():
        person_detector.detect_people()
        rospy.sleep(0.017)
    

