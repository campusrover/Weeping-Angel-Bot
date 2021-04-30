#!/usr/bin/env python

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

"""
NOTES:
1) Obstacle avoidance is implemented, but will need to have "refind person" mode implemented
2) Need to integrate person and face detection code into program
"""


class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

        #Change this subscriber to whatever is needed for person rec
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)

        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_cb)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.ranges = None
        self.twist = Twist()
        self.logcount = 0
        self.control = 0
        self.face = False

        #Temperoray Face array for testing
        #Change to TRUE if you want to test stop
        self.face_test = [False]

        self.last_image = None
        self.last_side = False

    def scan_cb(self, msg, cone=15):
        if self.face:
            self.go(0, speedlin=0)
            print("HUMAN!!!!")
        else:
            self.ranges =  np.array(msg.ranges)
        
            temp_speed = 0
            #limits the ranges between 0 and 10
            np.clip(self.ranges, 0, 10, out=self.ranges)

            #Reformating the range data
            left = self.ranges[360 - 4*cone:360 - cone]
            front = self.ranges[360 - cone:] + self.ranges[:cone]
            right = self.ranges[cone:4*cone]
            direction = int(np.amin(right) - np.amin(left)) + 1

            if np.amin(front) < 0.5:
                temp_speed += -float(direction) * np.pi/2
            elif np.amin(front) < 2:
                temp_speed += -float(direction) * 2

            print(temp_speed)
            self.go(self.control + temp_speed)

    #Placeholder for person and face detect code
    #
    #
    #
    #----------vvvvv--------
    def image_callback(self, msg):

        # get image from camera
        image = self.bridge.imgmsg_to_cv2(msg)

        # filter out everything that's not yellow
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = numpy.array([ 40, 0, 0])
        upper_yellow = numpy.array([ 120, 255, 255])
        mask = cv2.inRange(hsv,  lower_yellow, upper_yellow)
        masked = cv2.bitwise_and(image, image, mask=mask)

    # clear all but a 20 pixel band near the top of the image
        h, w, d = image.shape
        search_top = 3 * h /4
        search_bot = search_top + 20
        mask[0:search_top, 0:w] = 0
        mask[search_bot:h, 0:w] = 0
        cv2.imshow("band", mask)

    # Compute the "centroid" and display a red circle to denote it
        M = cv2.moments(mask)
        self.logcount += 1
        print("M00 %d %d" % (M['m00'], self.logcount))
    
        temp = []

        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)

            temp.append([cx, cy])
            
        self.centroid_processor(temp, image.shape, self.face_test)
        cv2.imshow("image", image)
        cv2.waitKey(3)
        print(self.control)
    #---------^^^^^^^^-----------
    #
    #
    #----------------------------


    
    #Processes all incoming image data, so to be used later on
    def centroid_processor(self, centroids, resolution, face_there):
        self.face = np.any(face_there)
        h, w, d = resolution
        err = 0
        if len(centroids) != 0:
            target = 0
            for x in range(0, len(centroids)):
                if centroids[x][1] < centroids[target][1]:
                    target = x
            err = centroids[target][0] - w/2
            err = -float(err) / 1000
            self.last_side = centroids[target][0] >= w/2
        

        self.control =  err

    def go(self, speedang, speedlin=0.2):
        self.twist = Twist()
        self.twist.linear.x = speedlin
        self.twist.angular.z = speedang
        self.cmd_vel_pub.publish(self.twist)
    
    

rospy.init_node('follower')

follower = Follower()
rospy.spin()