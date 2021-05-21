import pyautogui
import cv2
from collections import defaultdict
from pathlib import Path 
from scipy.signal import convolve2d
import math 
import numpy as np
from numpy import linalg as LA  
import moviepy.video.io.VideoFileClip as mpy 
import moviepy.editor as mp
d = Path(__file__).resolve().parents[1]
from sklearn.metrics import mean_squared_error
import pickle

with open(str(d) + "/HW4/part2_vectors", "rb") as f:
    part2 = pickle.load(f)


with open(str(d) + "/HW4/part3_vectors", "rb") as f:
    part3 = pickle.load(f)

with open(str(d) + "/HW4/part3_corrected_vectors", "rb") as f:
    part3_correct = pickle.load(f)
count = 0

for i in range(len(part2)):
    loss = mean_squared_error(part2[i], part3[i])
    loss_2 = mean_squared_error(part2[i], part3_correct[i])
    print("Turn_" + str(i + 1))
    print(loss)
    print(loss_2)
    if loss_2 <= loss:
        count = count+1
print(count)

