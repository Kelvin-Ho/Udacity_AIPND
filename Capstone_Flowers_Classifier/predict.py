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

from image import pretrained_model, load_checkpoint, process_image, predict, check_predict

import argparse
import json

def main():
    #path = "flowers\\valid\\9\\image_06414.jpg"
    #print(path)
    #Image.open(path)
    in_arg = get_input_args()
    
    arch = in_arg.arch
    checkpoint_path = in_arg.checkpoint
    top_k = in_arg.top_k
    #cat_to_name = in_arg.category_names
    image_path = in_arg.image_path
    #processing_unit = in_arg.gpu
    if torch.cuda.is_available() and in_arg.gpu=='gpu':
        print('GPU will be used')
        processing_unit = 'gpu'
    elif torch.cuda.is_available() == False:
        print('CPU will be used')
        processing_unit = 'cpu'


    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
 
    #print(in_arg,arch,checkpoint_path,top_k,cat_to_name,processing_unit,image_path)
    
    class_to_idx = torch.load(checkpoint_path)['class_to_idx']
    
    model_new = pretrained_model(arch)
    model = load_checkpoint(checkpoint_path,model_new)
    image = process_image(image_path)
    pros, indexes = predict(image, model, top_k, processing_unit)
    check_predict(class_to_idx, cat_to_name, pros, indexes, top_k)
    

def get_input_args():
    parser = argparse.ArgumentParser()
    
    # Argument 1: pretrained model arch
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'pre-trained model architecture') 
    
    # Argument 1: checkpoint path
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', 
                    help = 'checkpoint path') 
    
    # Argument 2: k number of top probabilities
    parser.add_argument('--top_k', type = int, default = '3', 
                    help = 'k number of top probabilities') 

    # Argument 3: category to names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'category to names') 
  
    # Argument 4: gpu
    parser.add_argument('--gpu', type = str, default = 'gpu', 
                    help = 'use gpu to train') 
   
    # Argument 5: image path
    path = "flowers/valid/9/image_06414.jpg"
    parser.add_argument('--image_path', type = str, default = path, 
                    help = 'image path') 
    
    return parser.parse_args()   
    
main()