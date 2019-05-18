#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image
from model import load_datas, classifier, train_model, pretrained_model, valid_model, save_checkpoint

import argparse

def main():
    in_arg = get_input_args()
    data_dir = in_arg.data_dir
    save_dir = in_arg.save_dir
    arch = in_arg.arch
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs
    #processing_unit = in_arg.gpu
    if torch.cuda.is_available() and in_arg.gpu=='gpu':
        print('GPU will be used')
        processing_unit = 'gpu'
    elif torch.cuda.is_available() == False:
        print('CPU will be used')
        processing_unit = 'cpu'
        
    print(in_arg)
    
    training_dataloaders, validation_dataloaders, testing_dataloaders,class_to_idx = load_datas(data_dir)
    pre_model = pretrained_model(arch)
    model = classifier(pre_model,hidden_units)
    after_train_model = train_model(model, training_dataloaders, validation_dataloaders,learning_rate,epochs,processing_unit)
    valid_model(after_train_model,testing_dataloaders,processing_unit)
    
    save_checkpoint(model,save_dir,class_to_idx)
    #train_model(model_1, training_dataloaders, validation_dataloaders)
    #valid_model(model,testing_dataloaders)
    
    #print(load_datas('flowers'))

    
    
    
def get_input_args():
    parser = argparse.ArgumentParser()
    
    # Argument 1: a path to a folder
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                    help = 'path to the folder of data') 
    
    # Argument 2: save directory
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                    help = 'save directory') 
    
    # Argument 3: pre-trained model architecture
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'pre-trained model architecture') 
    
    # Argument 4: learning rate
    parser.add_argument('--learning_rate', type = float, default = '0.001', 
                    help = 'learning rate') 
    
    # Argument 5: hidden units
    parser.add_argument('--hidden_units', type = int, default = '4096', 
                    help = 'hidden units') 
    
    # Argument 6: epochs
    parser.add_argument('--epochs', type = int, default = '2', 
                    help = 'number of epochs') 
    
    # Argument 7: gpu
    parser.add_argument('--gpu', type = str, default = 'gpu', 
                    help = 'use gpu to train') 
    
    return parser.parse_args()    


main()