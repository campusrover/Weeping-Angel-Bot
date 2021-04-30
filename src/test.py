#!/usr/bin/env python

import rospy, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

"""
PLEASE IGNORE THIS CODE:

THIS IS A DUMPING BLOCK OF CODE FOR PERSONAL TESTING AND RECORDING
"""

class Follower:
    def __init__(self):
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        self.logcount = 0


    def image_callback(self, msg):
        self.logcount += 1
        print(self.logcount)

    #Processes all incoming image data, so to be used later on
    def centroid_processor(self, centroids, resolution):
        h, w, d = resolution
        err = 0
        if centroids.size != 0:
            target = numpy.amin(centroids, axis=0)[1]
            err = centroid[target][0] - w/2
            err = -float(err) / 1000
            self.last_side = centroid[target][0] >= w/2
        return err
        

    def object_avoid(self, ranges):




rospy.init_node('follower')

follower = Follower()
rospy.spin()