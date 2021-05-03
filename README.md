# Weeping Angel Bot in Gazebo

![alt text](https://images.amcnetworks.com/bbcamerica.com/wp-content/uploads/2012/08/weepingangels.jpg)![alt text]()

## Overview

If you have ever watched the beloved sci-fi television show Doctor Who, you already know exactly what this robot does! But if you have not seen the show,
I will provide a brief explanation of the functionality of the robot.

Or goal for this project was to recreate the behavior of the Weeping Angel from Doctor Who. The Weeping Angel monster is a large stone statue of an angel-like creature which
looks relatively inconspicuous at first glance. However, when one looks away from this statue, it will come to life and chase the victim down, sending it back in time when it
reaches them. While inventing a time machine for this project proved to be a bit harder than we thought, we were able to implement physical the behavior of the Weeping Angel 
monster in the setting of a Gazebo simulation with a humanoid model and a Turtlebot3 Waffle simulated robot. 

In the simulation, we have a Turtlebot3 Waffle robot with a camera
on the font of it, and a fully mobile and animated humanoid model. The human model walks in a predifined path, while the robot follows it around the room. A ROS node subscribes
to the image topic produced by the robot's camera, and processes it using a convolutional neural network to detect the position of faces and people in the image. If the processing
node detects a person, but not a face in the robot's camera image it will publish cmd_vel topics to the robot that tell it to follow the person. If the node detects a face in the
camera image, it will publish cmd_vel commands telling the robot to stop in place. If the robot detects no people or faces in the camera image, it will publish commands that tell
the robot to wander around the map as if to search for new victims.
