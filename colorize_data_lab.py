from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os

from torchvision.transforms.functional import resize

import glob
import cv2


class ColorizeData(Dataset):
    def __init__(self, path):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256))
                                           ])
        
        self.show_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256))
                                           ])

        self.files_list = glob.glob(path+"*")

    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.files_list)
        
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        img_filename = self.files_list[index]

        image_bgr = cv2.imread(img_filename) #.astype("float32") 
        image = image_bgr
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        image_lab = image_lab.astype('float32')
        image_ab = (image_lab[:, :, 1:3]-128) /128.
        image_L = (image_lab[:, :, 0]-128)/128.
        image_rgb = (image.astype("float32")-128)/128.
        image_bgr = (image_bgr.astype("float32")-128)/128.

        image_in = self.input_transform(image_L)
        image_target = self.target_transform(image_ab)
        image_rgb = self.show_transform(image_rgb)
        image_bgr = self.show_transform(image_bgr)


        return (image_in, image_target, image_rgb)


def load_gray_image(path):
    '''
    Given path of image, load image and return it. 
    This function expects gray-scale image. 
    If input image is not gray-scale, it will transform it into gray. 

    Args:
        path: string, path to gray image
    Return:
        image: tensor, image in shape (1, h, w)
    '''

    if not os.path.exists(path):
        raise Exception('[ERROR]: Image not found: ' + path)

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image = image.astype('float32') 
    
    image = (image[:, :, 0] - 128.)/128.

    transform = T.Compose([T.ToTensor(),
                T.Resize(size=(256,256))
                ])

    image = transform(image)
    return image.unsqueeze(0)


    

        
