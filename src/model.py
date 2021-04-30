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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import pickle
import os
from time import time


class FaceRCNN:


    def load_model_txt(self, model, path):
        data_dict = {}
        fin = open(path, 'r')
        i = 0
        odd = 1
        prev_key = None
        while True:
            s = fin.readline().strip()
            if not s:
                break
            if odd:
                prev_key = s
            else:
                print('iter{}'.format(i))

                val = eval(s)
                if type(val) != type([]):
                    data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
                else:
                    data_dict[prev_key] = torch.FloatTensor(eval(s))
                i += 1
            odd = (odd + 1) % 2

        # Replace existing values with loaded

        print('Loading...')
        own_state = model.state_dict()
        print('Items:', len(own_state.items()))
        for k, v in data_dict.items():
            if not k in own_state:
                print('Parameter', k, 'not found in own_state!!!')
            else:
                try:
                    own_state[k].copy_(v)
                except:
                    print('Key:', k)
                    print('Old:', own_state[k])
                    print('New:', v)
                    sys.exit(0)
        print('Model loaded')
        torch.save(model.state_dict(), 'src/Term_Project/torch_model/model_state_dict.pt')


    def __init__(self):

        # print(os.getcwd())

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print("Model is running on {}".format(self.device))
        
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


