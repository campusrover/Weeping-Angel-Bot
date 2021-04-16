import numpy as np 
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class faceDetector:

    def __init__(self):

        img_sub = 