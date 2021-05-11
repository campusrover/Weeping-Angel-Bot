#!/usr/bin/env python

# Must run the following command in your workspace to install torch, torchvision, and future
# pip install torch==1.4.0 torchvision==0.5.0
# pip install future

# Additionally, this model requires you to have the path "../torch_model/model_state_dict.pth"
# The file for the state dictionary for the model exceeds github's file size limit, so go to the following
# link to download the state dictionary. Make sure it is put in a folder named "torch_model" in the main folder
# for this project.
# Link: https://drive.google.com/file/d/1n1nBDpdu9GnAb006depSl32x6O47NU_D/view?usp=sharing 







import torch
import torchvision
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection.rpn import AnchorGenerator
import numpy as np
import pickle
import os
from time import time
import io

from mobilenetv3 import mobilenetv3_small



class FaceRCNN:


    def __init__(self, mobile_net=True):

        # print(os.getcwd())

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print("Model is running on {}".format(self.device))

        if mobile_net:

            backbone = mobilenetv3_small()
            # state_dict = torch.load('src/Term_Project/torch_model/mobilenetv3smallpretrained.pth')
            # backbone.load_state_dict(state_dict)
            backbone = backbone.features

            backbone.out_channels=96
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
            model = FasterRCNN(backbone,
                   num_classes=3,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
            state_dict = torch.load('src/Term_Project/torch_model/mobilenet_v3_state_dict.pth')
            model.load_state_dict(state_dict)
            self.model = model
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

            # Load the pretrained weights of the model from the downloaded file.
            self.model.load_state_dict(torch.load('src/Term_Project/torch_model/model_state_dict.pth'))

        self.model.to(self.device)
        self.model.eval()
        
        self.labels_dict =  {1:'Person', 2:'Face'}
        self.colors_dict = {1:(0,255,0), 2:(0,0,255)}

    # Prints the number of parameters in the model to the terminal
    def get_n_params(self):
        pp=0
        for p in list(self.model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        print(pp)

    def simple_detection(self, img):
        # Preprocessing the input image
        start = time()
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor_normal = image_tensor/255
        duration = time() - start
        print("\nPreprocessing: {} s".format(duration))

        # Passing preprocessed image into model
        start = time()
        output = self.model([image_tensor_normal])
        duration = time() - start
        print("\nModel processing: {} s".format(duration))

        # Returns the bounding boxes, labels for each bounding box, and the prediction probabilities for each box
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        return boxes.detach().numpy(), labels.detach().numpy(), scores.detach().numpy()


