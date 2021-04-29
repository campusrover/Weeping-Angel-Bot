#!/usr/bin/env python

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class Follower:
    def __init__(self):
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        self.logcount = 0


    def image_callback(self, msg):
        self.logcount += 1
        print(self.logcount)


rospy.init_node('follower')

follower = Follower()
rospy.spin()