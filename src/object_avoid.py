#!/usr/bin/env python

import rospy, numpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import random
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool

class Follower:
    def __init__(self):
        self.person = rospy.Subscriber('/person_centroid', Int32MultiArray, self.centroid_cd)
        self.face_bool = rospy.Subscriber('/face_detected', Bool, self.face_cd)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_cb)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.ranges = None
        self.twist = Twist()
        self.control = 0
        self.centroid_data = np.array([-1, -1])
        self.face = False

    def scan_cb(self, msg, cone=100):
        if False:#self.face:
            self.go(0, speedlin=0)
            print("HUMAN SEES ME!!!!")
        else:
            self.ranges =  np.array(msg.ranges)

            temp_speed = 0
            #limits the ranges between 0 and 10
            np.clip(self.ranges, 0, 10, out=self.ranges)

            #Reformating the range data
            right = np.min(self.ranges[360 - cone: 360 - cone/6])
            left = np.min(self.ranges[cone/6 :cone])
            front = min(np.min(self.ranges[360 - cone/6:]), np.min(self.ranges[:cone/6]))

            direction = int(right - left) + 0.25

            if front < 1.5:
                if front < .75:
                    temp_speed += -float(direction) - self.control
                    print("CLOSE")
                
                temp_speed += -float(direction) + self.control
                print("AVOIDING")
            else:
                temp_speed = self.control
                print("FOLLOWING")

            print("Speed: ", temp_speed)
            self.go(temp_speed)
            self.control = 0

    def centroid_cd(self, centroids):
        self.centroid_data = np.array(centroids.data)
        resolution = [1080, 1920, 0]
        self.centroid_processor(self.centroid_data, resolution)
    
    def face_cd(self, Bool):
        self.face = Bool.data
    
    #Processes all incoming image data, so to be used later on
    def centroid_processor(self, centroids, resolution):
        h, w, d = resolution
        err = 0
        if not np.array_equal(centroids, [-1, -1]):
            print("Person location: ", centroids)
            err = centroids[0] - w/2
            err = -float(err) / 1000
        else:
            err = min(max(random.uniform(-10, 10), -1), 1)

        self.control =  err

    def go(self, speedang, speedlin=0.2):
        self.twist = Twist()
        self.twist.linear.x = speedlin
        self.twist.angular.z = speedang
        self.cmd_vel_pub.publish(self.twist)
    
    

rospy.init_node('follower')

follower = Follower()
rospy.spin()