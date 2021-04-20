#!/usr/bin/env python


import numpy as np 
import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

class faceDetector:

    def __init__(self):

        self.gray_img_sub = rospy.Subscriber('/gray_img', Image, self.gray_img_cb)

        self.color_img_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.color_img_cb)

        self.img_pub = rospy.Publisher('/face_img', Image, queue_size=1)

        self.face_detected_pub = rospy.Publisher('/face_detected', Bool, queue_size=1)

        self.face_detected = Bool()

        self.bridge = CvBridge()

        self.face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')


    def gray_img_cb(self, img):
        self.gray_img = self.bridge.imgmsg_to_cv2(img)

    def color_img_cb(self, color_img):
        self.color_img = self.bridge.imgmsg_to_cv2(color_img, 'bgr8')



    def detect_faces(self):
        faces = self.face_cascade.detectMultiScale(self.gray_img, scaleFactor = 1.1, minNeighbors = 3, minSize = (70,70) )
        for (x, y, w, h) in faces:
            cv2.rectangle(self.color_img,(x,y),(x+w,y+h),(255,0,0),2)
        if len(faces) > 0:
            self.face_detected.data = True
        else:
            self.face_detected.data = False
        self.face_detected_pub.publish(self.face_detected)
        img_msg = self.bridge.cv2_to_imgmsg(self.color_img, 'bgr8')
        self.img_pub.publish(img_msg)


if __name__ == "__main__":
    rospy.init_node('face_detector')
    face_detector = faceDetector()
    rospy.sleep(1)
    while not rospy.is_shutdown():
        face_detector.detect_faces()
        rospy.sleep(0.017)
    