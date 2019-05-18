#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image

def pretrained_model(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    return model

def load_checkpoint(checkpoint_file,model_new):
    checkpoint = torch.load(checkpoint_file)
    model_new.classifier = checkpoint['classifier']
    model_new.load_state_dict(checkpoint['state_dict'])
    model_new.class_to_idx = checkpoint['class_to_idx']
    return model_new

def process_image(image_path):
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
  
    #print(img)
    width, height = img.size
    #print(width, height)
    if width <= height:
        width, height = 256, int(256 / width * height)
    else:
        width, height = int(256 / height * width), 256
    #print(width, height)
    img = img.resize((width, height))
    #print(img.size)
    left = (width - 224) / 2 
    right = left + 224
    top = (height - 224) / 2
    bot = top + 224
    img = img.crop((left, top, right, bot))
    #print(img.size)
    np_img = np.array(img)
    #print(np_img.shape)
    #print(np_img[2])
    np_img = np_img / 255
    #print(np_img[2])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    #np_img = np.clip(np_img,0,1)
    np_img = np_img.transpose((2,0,1))
    
    np_img = torch.from_numpy(np_img).float()
    
    return np_img
    

def predict(process_image, model, top_k, processing_unit):
    model.eval()
    #device = torch.device('cuda:0' if (processing_unit=='gpu') else 'cpu')
    #model.cuda()
    if processing_unit=='gpu':
        model.cuda()
        device = 'cuda:0'
    elif processing_unit == 'cpu':
        model.cpu()
        device = 'cpu'
    # TODO: Implement the code to predict the class from an image file
    img_tensor = torch.from_numpy(np.expand_dims(process_image, axis=0)).to(device)
    outputs = model.forward(img_tensor)
    pros_log, indexes = outputs.topk(top_k)
    pros = torch.exp(pros_log)
    
    return pros.data.to('cpu').numpy(), indexes.data.to('cpu').numpy()


def check_predict(class_to_idx, cat_to_name, pros, indexes, top_k):
    idx_to_class = {idx:cls for cls, idx in class_to_idx.items()}

    #image_path = 'flowers/test/28/image_05214.jpg'

    #pros, indexs = predict(image_path, model, topk=5)

    dict = {}
    for i in range(top_k):
        max_ps = pros[0][i]
        max_class = idx_to_class[indexes[0][i]]
        #print(max_class)
        #print(cat_to_name)
        max_name = cat_to_name[max_class]
        dict[max_name] = max_ps
    print(dict)
    

    