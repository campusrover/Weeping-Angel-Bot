#!/usr/bin/env python

import numpy as np 
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class color2Gray:

    def __init__(self):

        self.img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_cb)

        self.img_pub = rospy.Publisher('/gray_img', Image, queue_size=1)

        self.bridge = CvBridge()


    def img_cb(self, img):
        
        cv_img = self.bridge.imgmsg_to_cv2(img, 'bgr8')
        gray_cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = self.bridge.cv2_to_imgmsg(gray_cv_img)
        self.img_pub.publish(gray_img)




if __name__ ==  "__main__":
    rospy.init_node('gray_img_pub')
    color_2_gray = color2Gray()
    rospy.spin