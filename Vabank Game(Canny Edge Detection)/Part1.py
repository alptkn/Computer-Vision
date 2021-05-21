# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:56:16 2020

@author: 90553
"""
import pyautogui
import time 
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

path = str(Path(__file__).parent)



def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range (image_row):
        for col in range (image_col):
            output[row,col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output


def gaussian_kernel(size, sigma=1):
    kernel = np.zeros((size,size))
    mean = size / 2
    sum = 0.0
    for i in range(size):
        for j in range(size):
            kernel[i][j] = np.exp(-0.5 * np.power((i-mean)/sigma, 2.0) + pow((j-mean) / sigma, 2.0))/(2 * np.pi * np.power(sigma,2))
            sum = sum + kernel[i][j]

    return (kernel / sum)

def gaussian_smooth(image, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolution(image, kernel)

def sobel_edge_detector(image,s_filter):
    image_x = convolution(image, s_filter)
    image_y = convolution(image, np.flip(s_filter.T, axis=0))
    gradient = np.sqrt(np.square(image_x) + np.square(image_y))
    gradient *= 255.0 / gradient.max()
    
    return gradient


    

time.sleep(5)
pyautogui.click(path + "/all_shapes_button.PNG")
screenShot = pyautogui.screenshot()



image = np.array(screenShot)
s_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
clone = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
out = gaussian_smooth(clone,3,1)
sobel_1 = sobel_edge_detector(clone,s_filter)
sobel_2 = sobel_edge_detector(out, s_filter)

cv2.imwrite( path + "/sobel_original_img.PNG", sobel_1)
cv2.imwrite(path + "/sobel_smooth_img.PNG", sobel_2)


high_thresh, thresh_im = cv2.threshold(clone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
lowThresh = 0.5*high_thresh

v = np.median(clone)
sigma = 1
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

canny = cv2.Canny(clone,lowThresh, high_thresh)
cv2.imwrite(path + "/canny_img.PNG", canny)





