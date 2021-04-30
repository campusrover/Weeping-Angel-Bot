#!/usr/bin/env python

# Must run the following command in your workspace to install torch, torchvision, and future
# pip install torch==1.4.0 torchvision==0.5.0
# pip install futuregf

# Additionally, this model requires you to have the path "../torch_model/model_state_dict.pth"

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
        print(self.device)
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        self.model.load_state_dict(torch.load('src/Term_Project/torch_model/model_state_dict.pth'))        
        # self.model = torch.load('src/Term_Project/torch_model/entire_model.pth')

        self.model.to(self.device)
        self.model.eval()
        
        self.labels_dict =  {1:'Person', 2:'Face'}
        self.colors_dict = {1:(0,255,0), 2:(0,0,255)}


    def get_n_params(self):
        pp=0
        for p in list(self.model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        print(pp)

    def simple_detection(self, img):
        start = time()
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        print(image_tensor.shape)
        image_tensor_normal = image_tensor/255
        duration = time() - start
        print("\nPreprocessing: {} s".format(duration))

        # try:
        start = time()
        output = self.model([image_tensor_normal])
        duration = time() - start
        print("\nModel processing: {} s".format(duration))
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        # inds_clean_scores = []
        # for i in range(scores.shape[0]):
        #     if scores[i] >= 0.8:
        #         inds_clean_scores.append(i)
        # inds_clean_scores = torch.LongTensor(inds_clean_scores, device=self.device)
        # clean_boxes = torch.index_select(boxes, 0, inds_clean_scores)
        # clean_labels = torch.index_select(labels, 0, inds_clean_scores)
        # return clean_boxes, clean_labels
        return boxes.detach().numpy(), labels.detach().numpy(), scores.detach().numpy()
        # except:
        #     pass




    def person_boxes(self, img):
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor_normal = image_tensor/255
        output = self.model([image_tensor_normal])
        print('processed the output')
        boxes = output[0]['boxes'].numpy()
        print(boxes)
        labels = output[0]['labels'].numpy()
        print(labels)
        scores = output[0]['scores'].numpy()
        print(scores)
        inds_clean_scores = []
        for i in range(scores.shape[0]):
            if scores[i] >= 0.8:
                inds_clean_scores.append(i)
        inds_clean_scores = torch.LongTensor(inds_clean_scores).to(self.device)
        clean_boxes = torch.index_select(boxes, 0, inds_clean_scores)
        clean_labels = torch.index_select(labels, 0, inds_clean_scores).tolist()
        clean_scores = torch.index_select(scores, 0, inds_clean_scores)
        image_tensor = torchvision.utils.draw_bounding_boxes(image_tensor.detach().cpu(),
                                                clean_boxes,
                                                [self.labels_dict[id] + ": {}%".format(str(round(float(scores[i])*100, 2))) for i, id in enumerate(clean_labels)],
                                                [self.colors_dict[id] for id in clean_labels],
                                                fill=False,
                                                width=2,
                                                font_size=15)
        image_tensor = torch.transpose(image_tensor, 2,1)
        image_tensor = torch.transpose(image_tensor, 2,0)
        return clean_boxes, clean_labels, image_tensor.numpy()
