import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.color import rgb2lab, lab2rgb
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        ''' Multiplying n to val before summation is done because
            it is usually used for loss which is already mean with respect to batch size. 
        '''
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def plot_samples_from_images_lab(L, ab, batch_size, plot_path, filename):
    ''' Plot images
    Changed 2020.11.23
    isRange01 is added to normalize image in different way. 

    Args: 
        L: (b, h, w), tensor L image 
        ab: (b, 2, h, w), tensor ab
        batch_size: int
        plot_path: string
        filename: string
    '''

    L = (L+1)/2.0 * 255.
    ab = (ab+1)/2 * 255.
    images = torch.cat((L, ab), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)
    images = images.astype(np.uint8)


    if batch_size > 1:
        fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
            image = cv2.cvtColor(images[idx], cv2.COLOR_LAB2BGR) #/ 255.0
            ax.imshow(image)
    else:
        fig = plt.figure(frameon=False)
        image = cv2.cvtColor(images[0], cv2.COLOR_LAB2BGR)
        plt.axis('off')
        plt.imshow(image)

    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    plt.savefig(os.path.join(plot_path, filename), bbox_inches='tight')
    plt.close()


def plot_samples_from_images(images, batch_size, plot_path, filename):
    ''' Plot images
    Changed 2020.11.23
    isRange01 is added to normalize image in different way. 

    Args: 
        images: (b, c, h, w), tensor in any range. (c=3 or 1)
        batch_size: int
        plot_path: string
        filename: string
    '''
    max_pix = torch.max(torch.abs(images))
    if max_pix != 0.0:
        images = ((images/max_pix) + 1.0)/2.0
    else:
        images = (images + 1.0) / 2.0
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1) 

    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    if batch_size > 1:
        fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
            ax.imshow(images[idx])
    else:
        fig = plt.figure(frameon=False)
        plt.axis('off')
        plt.imshow(images[0])

    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    plt.savefig(os.path.join(plot_path, filename), bbox_inches='tight')
    plt.close()
