#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class bridge:

    def __init__(self):

        self.bridge = CvBridge()

    

    def to_np(self, img):

        return self.bridge.imgmsg_to_cv2(img, 'bgr8')

    
    def to_msg(self, img):

        return self.bridge.cv2_to_imgmsg(img)
