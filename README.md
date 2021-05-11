# Weeping Angel Bot in Gazebo
## By Nathan Cai and Adam Ring

<img src="weepingangels.jpg"  width="325" height="250"/>         <img src="weepingangelGIF.gif"  width="500" height="250"/>

### See [Github Pages Site](https://aringan0323.github.io/weepingangelbot.github.io/) for more details on this project.

## Overview

If you have ever watched the beloved sci-fi television show Doctor Who, you already know exactly what this robot does! But if you have not seen the show,
I will provide a brief explanation of the functionality of the robot.

Our goal for this project was to recreate the behavior of the Weeping Angel from Doctor Who. The Weeping Angel monster is a large stone statue of an angel-like creature which
looks relatively inconspicuous at first glance. However, when one looks away from this statue, it will come to life and chase the victim down, sending it back in time when it
reaches them. While inventing a time machine for this project proved to be a bit harder than we thought, we were able to implement the physical behavior of the Weeping Angel 
monster in the setting of a Gazebo simulation with a humanoid model and a Turtlebot3 Waffle simulated robot. 

In the simulation, we have a Turtlebot3 Waffle robot with a camera
on the font of it, and a fully mobile and animated humanoid model. The human model walks in a predifined path, while the robot follows it around the room. A ROS node subscribes
to the image topic produced by the robot's camera, and processes it using a convolutional neural network to detect the position of faces and people in the image. If the processing
node detects a person, but not a face in the robot's camera image it will publish cmd_vel topics to the robot that tell it to follow the person. If the node detects a face in the
camera image, it will publish cmd_vel commands telling the robot to stop in place. If the robot detects no people or faces in the camera image, it will publish commands that tell
the robot to wander around the map as if to search for new victims.

## Installation (Old)

1. In order to run this program, you will need to be running an Ubuntu 18.04 environment with ROS Melodic installed. It is recommended to have CUDA 10.1 installed in order to run the image processing with GPU acceleration. Instructions for installing CUDA 10.1 on Ubuntu 18.04 can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. Open a terminal with a path to your catkin workspace src folder, and `git clone https://github.com/campusrover/Weeping-Angel-Bot.git`. Rename the folder created from `Weeping-Angel-Bot` to `Term_Project`.

3. `pip install torch torchvision` and then `pip install future`.

4. Create a new folder in your `Term_Project` folder and name it `torch_model`

5. Download the [model weights](https://drive.google.com/file/d/1n1nBDpdu9GnAb006depSl32x6O47NU_D/view) for the neural network. Move `model_state_dict.pth` into the `torch_model` folder.

6. Return to your `catkin_ws` folder and call `catkin_make` in the terminal.

## Installation (Updated)


1. In order to run this program, you will need to be running an Ubuntu 18.04 environment with ROS Melodic installed. It is recommended to have CUDA 10.1 installed in order to run the image processing with GPU acceleration. Instructions for installing CUDA 10.1 on Ubuntu 18.04 can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. Open a terminal with a path to your catkin workspace src folder, and `git clone https://github.com/campusrover/Weeping-Angel-Bot.git`. Rename the folder created from `Weeping-Angel-Bot` to `Term_Project`.

3. Run the commands `pip install torch torchvision`, `pip install future` and then `pip install typing`.

4. Create a new folder in your `Term_Project` folder and name it `torch_model`

5. Download the [model weights](https://drive.google.com/file/d/1Z896xM4eIf6pd-g54yUqheyyl4G7r8A_/view?usp=sharing) for the neural network. Move `mobilenet_v3_state_dict.pth` into the `torch_model` folder.

6. Return to your `catkin_ws` folder and call `catkin_make` in the terminal.

## Usage

1. Navigate to your `catkin_ws` folder in a terminal and run the command `roslaunch Term_Project weeping_angel.launch`. The gazebo simulation should launch.

2. In a different terminal with the same path run the command `rosrun Term_Project pytorch_detection.py` to start the neural network detector node.

3. Finally run the command `rosrun Term_Project object_avoid.py` to start the tracking

## How It Works

### Person/Face Detection Technology

#### Model Training (Old)

* The person and face detection in this program is performed by a [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf) with a [ResNet-50](https://arxiv.org/pdf/1512.03385.pdf) as the backbone convolutional network.
* The FasterRCNN is pretrained on the [COCO](https://cocodataset.org/#home) dataset, which contains thousands of images of objects from 80 different categories. Each object of a category in each image is annotated with a segmentation mask and a bounding box, as well as the category that it fits into. 
* The FasterRCNN was fine-tuned on the [faces4coco](https://github.com/ACI-Institute/faces4coco) dataset, which annotates all of the COCO images with bounding boxes of only people and faces.
* The FasterRCNN was trained on the validation set of the faces4coco dataset over 10 epochs.
* System Specs
  - AMD Ryzen9 3950x
  - RTX 2070 Super
  - 32GB DDR4
  - 2TB Nvme M.2 SSD

#### Model Training (Updated)

* The person and face detection in this program is performed by a [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf) with a [MobileNet V3](https://arxiv.org/pdf/1905.02244.pdf) as the backbone convolutional network.
* The FasterRCNN is pretrained on the [COCO](https://cocodataset.org/#home) dataset, which contains thousands of images of objects from 80 different categories. Each object of a category in each image is annotated with a segmentation mask and a bounding box, as well as the category that it fits into. 
* The FasterRCNN was fine-tuned on the [faces4coco](https://github.com/ACI-Institute/faces4coco) dataset, which annotates all of the COCO images with bounding boxes of only people and faces.
* The FasterRCNN with the MobileNet V3 backbone was trained on the validation set of the faces4coco dataset over 30 epochs.
* System Specs
  - AMD Ryzen9 3950x
  - RTX 2070 Super
  - 32GB DDR4
  - 2TB Nvme M.2 SSD

#### Model Deployment

* In the program, the model is loaded from pytorch and then the pretrained weights are imported from `model_state_dict.pth`.
* The detection node subscribes to the image topic published by the robot's camera, and then the image is passed through the model. The model outputs 3 arrays which contain bounding boxes of the objects in the image, prediction scores for those objects, and then category labels for the objects detected.
* If the model detects any people in the image, then the detection node will select the person detected with the highest prediction score and then calculate the center point of the bounding box and publish it as a topic.
* If the model detects any faces in the image, then the detection node will publish a boolean topic named `face_detected` as `True`. Otherwise, it will publish `face_detected` as `False`.

### Object Avoidance

#### Person Following

* The person following in this program uses the image location data provided by the Person/Face detection code.
* The following code takes in the coordinate data of the person center point and compare it to is x-position in the image. Based off the center point's offset from the image's center point, the code is able to convert it into a angular velocity that is publised to the robot.
* The following code also takes in face detection data from the Person/Face detection code. If the Person/Face detection code detects a face, the robot will come to an immidate stop and remain immobile till the Person/Face detection code no longer detects a face
* If the person is no longer visable to the robot, the robot will go into a random wandering mode in an attempt to relocate the person.

#### Obstacle Avoidance

* The obstacle avoidance in this program uses the LIDAR data published by the robot's on-board LIDAR.
* The program checks to see if there are any obstacles in front of the robot and compensates according.
* This is achieved by using the LIDAR data to check the forward, left, and right regions of the robot. If an obstacle is detected in the forwards region, based off of the left and right data, the robot will gradually turn to try and avoid the object.
* If the program is also designed to accomodate the tacking data too. As both the object avoidance adjustments works in tandem to influence the robot, this is broken only if the robot's trajectory is within too close of proximity to an obstacle.

### Control Flow

1. The robot subscribes to the Person/Face detection code
2. The robot recieves LIDAR data about the area
3. The Preson/Face detections returns person location
4. The Stops if face is detected, else it will start detecting obstacles
5. If an obstacle is found, it will turn to avoid the object while trying to maintain view of the person, however, if the robot is too close and is in danger of a collision, the robot will ignore person location and turn
6. Repeat

## Licence

The MIT License (MIT) 

## Links to Lab Notebook Entries

- Adam Ring: [Saving and Loading Pytorch Models](https://campus-rover.gitbook.io/lab-notebook/advanced-topics/computer-vision/pretrained-model-deployment)
- Nathan Cai: [Spawning Animated Human](https://campus-rover.gitbook.io/lab-notebook/faq/spawning_animated_human)



