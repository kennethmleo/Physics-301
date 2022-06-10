import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.models as models

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
from PIL import Image
from Activation_maps import visualize_activation_maps

transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

alexnet = models.alexnet(pretrained = True)

with open('class_names_ImageNet.txt') as labels:
    classes = [i.strip() for i in labels.readlines()]
    
def Object_detection_alexnet(image):
    if type(image) == str: img = Image.open(image)
    else: img = Image.fromarray(image)
        
    transformed_img = transform_image(img)
    batch_img = torch.unsqueeze(transformed_img,0)

    output = alexnet(batch_img)
    
    sorted_, indices = torch.sort(output, descending = True)
    percentage = F.softmax(output, dim = 1)[0] * 100.0
    class_ = [classes[i] for i in indices[0][:5]]
    percent_ = np.array([percentage[i].item() for i in indices[0][:5]])
    percent_ = np.round(percent_, 2)
        
    return class_, percent_