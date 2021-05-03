# Weeping Angel Bot in Gazebo

<img src="weepingangels.jpg"  width="325" height="250"/>         <img src="weepingangelGIF.gif"  width="500" height="250"/>

## Overview

If you have ever watched the beloved sci-fi television show Doctor Who, you already know exactly what this robot does! But if you have not seen the show,
I will provide a brief explanation of the functionality of the robot.

Or goal for this project was to recreate the behavior of the Weeping Angel from Doctor Who. The Weeping Angel monster is a large stone statue of an angel-like creature which
looks relatively inconspicuous at first glance. However, when one looks away from this statue, it will come to life and chase the victim down, sending it back in time when it
reaches them. While inventing a time machine for this project proved to be a bit harder than we thought, we were able to implement the physical behavior of the Weeping Angel 
monster in the setting of a Gazebo simulation with a humanoid model and a Turtlebot3 Waffle simulated robot. 

In the simulation, we have a Turtlebot3 Waffle robot with a camera
on the font of it, and a fully mobile and animated humanoid model. The human model walks in a predifined path, while the robot follows it around the room. A ROS node subscribes
to the image topic produced by the robot's camera, and processes it using a convolutional neural network to detect the position of faces and people in the image. If the processing
node detects a person, but not a face in the robot's camera image it will publish cmd_vel topics to the robot that tell it to follow the person. If the node detects a face in the
camera image, it will publish cmd_vel commands telling the robot to stop in place. If the robot detects no people or faces in the camera image, it will publish commands that tell
the robot to wander around the map as if to search for new victims.

## Installation

1. In order to run this program, you will need to be running an Ubuntu 18.04 environment with ROS Melodic installed. It is recommended to have CUDA 10.1 installed in order to run the image processing with GPU acceleration. Instructions for installing CUDA 10.1 on Ubuntu 18.04 can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. Open a terminal with a path to your catkin workspace src folder, and `git clone https://github.com/campusrover/Weeping-Angel-Bot.git`. Rename the folder created from `Weeping-Angel-Bot` to `Term_Project`.

3. `pip install torch torchvision` and then `pip install future`.

4. Create a new folder in your `Term_Project` folder and name it `torch_model`

5. Download the [model weights](https://drive.google.com/file/d/1n1nBDpdu9GnAb006depSl32x6O47NU_D/view) for the neural network. Move `model_state_dict.pth` into the `torch_model` folder.

6. Return to your `catkin_ws` folder and call `catkin_make` in the terminal.

## Usage

1. Navigate to your `catkin_ws` folder in a terminal and run the command `roslaunch Term_Project weeping_angel.launch`. The gazebo simulation should launch.

2. In a different terminal with the same path run the command `rosrun Term_Project pytorch_detection.py` to start the neural network detector node.

3. Finally ***add stuff here for object avoidance***

## How It Works

### Person/Face Detection Technology

* The person and face detection in this program is performed by a [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf) with a [ResNet-50](https://arxiv.org/pdf/1512.03385.pdf) as the backbone convolutional network.
* The FasterRCNN is pretrained on the [COCO](https://cocodataset.org/#home) dataset, which contains thousands of images of objects from 80 different categories. Each object of a category in each image is annotated with a segmentation mask and a bounding box, as well as the category that it fits into. 
* The FasterRCNN was fine-tuned on the [faces4coco](https://github.com/ACI-Institute/faces4coco) dataset, which annotates all of the COCO images with bounding boxes of only people and faces.
* The FasterRCNN was trained on the validation set of the faces4coco dataset over 10 epochs on a RTX 2070 Super.

### Object Avoidance

### Control Flow

## Licence

The MIT License (MIT) 



