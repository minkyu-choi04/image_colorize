import os
import colorize_data_lab as D
import argparse
import misc
import time
from scipy.interpolate import UnivariateSpline
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Color Temerature')

parser.add_argument('--path-img', default='test_img.jpg', 
        type=str, help='path to image to control color temperature')


def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))

def color_warm(image):
    '''
    Control color of image to be warm
    Reference: OpenCV with Python Blueprints

    Args:
        image: image to be transformed
    Return: 
        img_bgr_warm: temperature controlled image
    '''
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
            [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
            [0, 30, 80, 120, 192])

    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.merge((c_b, c_g, c_r))

    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    # increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm,
        cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

    img_bgr_warm = cv2.cvtColor(cv2.merge(
        (c_h, c_s, c_v)),
        cv2.COLOR_HSV2BGR)
    return img_bgr_warm

def color_cool(image):
    '''
    Control color of image to be cool
    Reference: OpenCV with Python Blueprints

    Args:
        image: image to be transformed
    Return: 
        img_bgr_warm: temperature controlled image
    '''
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
            [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
            [0, 30, 80, 120, 192])
    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img_bgr_cold = cv2.merge((c_b, c_g, c_r))

    # decrease color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_cold,
        cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
    img_bgr_cold = cv2.cvtColor(cv2.merge(
        (c_h, c_s, c_v)),
        cv2.COLOR_HSV2BGR)
    return img_bgr_cold

def save_image(path, filename, image):
    '''
    Save image into jpg file. 

    Arge:
        path: path to save dir.
        filename: name of file to be saved.
        image: numpy array to be saved.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig = plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(image)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')
    plt.close()


def main():
    args = parser.parse_args()

    if not os.path.isdir('./results'):
        os.makedirs('./results')
    
    img = cv2.imread(args.path_img)

    img_warm = color_warm(img)
    img_cool = color_cool(img)

    
    fn_warm = args.path_img.split('.')
    fn_warm[0] = fn_warm[0] + '_warm'
    fn_cool = args.path_img.split('.')
    fn_cool[0] = fn_cool[0] + '_cool'

    save_image('./results', fn_warm[0], img_warm)
    save_image('./results', fn_cool[0], img_cool)



if __name__ == "__main__":
    main()
