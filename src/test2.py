#!/usr/bin/env python

import rospy, numpy, cv2, cv_bridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import random
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool

class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_cb)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.ranges = None
        self.twist = Twist()
        self.control = 0
        self.centroid_data = np.array([-1, -1])
        self.face = False

    def scan_cb(self, msg, cone=90):
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

    def centroid_cd(self, centroids):
        self.centroid_data = np.array(centroids.data)
        resolution = [1080, 1920, 0]
        self.centroid_processor(self.centroid_data, resolution)

    def image_callback(self, msg):

        # get image from camera
        image = self.bridge.imgmsg_to_cv2(msg)

        # filter out everything that's not yellow
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = numpy.array([ 36, 0, 0])
        upper_yellow = numpy.array([ 70, 255, 255])
        mask = cv2.inRange(hsv,  lower_yellow, upper_yellow)
        masked = cv2.bitwise_and(image, image, mask=mask)

        h, w, d = image.shape
        cv2.imshow("band", mask)

    # Compute the "centroid" and display a red circle to denote it
        M = cv2.moments(mask)

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)
            self.centroid_data = np.array([cx, cy])
            resolution = image.shape
            self.centroid_processor(self.centroid_data, resolution)

        cv2.imshow("image", image)
        cv2.waitKey(3)
    
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
            err = min(max(random.uniform(-10, 10), -6), 6)

        self.control =  err

    def go(self, speedang, speedlin=0.2):
        self.twist = Twist()
        self.twist.linear.x = speedlin
        self.twist.angular.z = speedang
        self.cmd_vel_pub.publish(self.twist)
    
    

rospy.init_node('follower')

follower = Follower()
rospy.spin()