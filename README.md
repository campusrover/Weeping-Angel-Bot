# Weeping Angel Bot in Gazebo

<img src="weepingangels.jpg"  width="325" height="250"/>         <img src="weepingangelGIF.gif"  width="500" height="250"/>

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

## Installation

* In order to run this program, you will need to be running an Ubuntu 18.04 environment with ROS Melodic installed. It is recommended to have CUDA 10.1 installed in order to run the image processing with GPU acceleration.
* Open a terminal with a path to your catkin workspace src folder, and `git clone https://github.com/campusrover/Weeping-Angel-Bot.git`. Rename the folder created from `Weeping-Angel-Bot` to `Term_Project`.
* `pip install torch torchvision` and then `pip install future`.
* Create a new folder in your `Term_Project` folder and name it `torch_model`
* Download the [model weights]https://drive.google.com/file/d/1n1nBDpdu9GnAb006depSl32x6O47NU_D/view for the neural network. Move `model_state_dict.pth` into the `torch_model` folder.


