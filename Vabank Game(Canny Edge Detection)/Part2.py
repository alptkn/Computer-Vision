import pyautogui
import time 
import math
import numpy as np
from scipy import signal as sig
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
from skimage.feature import corner_peaks

path = str(Path(__file__).parent)



def gradients(image):
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.flip(kernel_x.T)
    
    return sig.convolve2d(image, kernel_x, 'same'), sig.convolve2d(image, kernel_y, 'same')

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
    return sig.convolve2d(image, kernel,'same')




def find_harris_corners(img, window_size=5, k=0.04, thresh=10000):
    
    dy, dx = np.gradient(img)
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy
    
    height = img.shape[0]
    width = img.shape[1]
    
    clone = img.copy()
    color_img = cv2.cvtColor(clone, cv2.COLOR_GRAY2RGB)
    offset = int(window_size / 2)
    
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            
            """
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]"""
            
            
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            
            
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            
            
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            R = det - k * (trace**2)
            
            if R > thresh:
                color_img[y,x] = (0,0,255)
        
    return color_img
            
            


image = cv2.imread(path + "/Ayıp.PNG")
cv2.imshow("Figure_1", image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corner_img = find_harris_corners(gray, 2, 0.04,10000)


cv2.imwrite(path + "/Ayıp_artık.PNG", corner_img)





