import pyautogui
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
import argparse
path = str(Path(__file__).parent)


parser = argparse.ArgumentParser()
parser.add_argument('--game', type=str, default="Vabank", help="Name of game (Vabank, Shame")
opt = parser.parse_args()

game_type = opt.game

if game_type == "Vabank":
    name = str("/Vabank_shapes/shape_")
    button = "/Vabank_button.PNG"
elif game_type == "Shame":
    name = str("/Shame/Shape_")
    button = "/Shame_button.PNG"



key_list = []

for i in range(18):
    img_path = path + name + str(i+1) + ".PNG"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,3,0.04)
    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    if len(corners) - 1 == 3:
        key_list.append('A')
    elif len(corners) - 1 == 4:
        key_list.append('S')
    elif len(corners) - 1 == 10:
        key_list.append('D')
    elif len(corners) - 1 == 6:
        key_list.append('F')
    else:
        key_list.append('x')
        print("Image cannot be recognized")


time.sleep(5)
pyautogui.click(path + button)
count = 0
while True:
    
    if pyautogui.pixel(661,672)[0] == 0:
        if key_list[count] == 'A':
            pyautogui.press('a')
        elif key_list[count] == 'S':
            pyautogui.press('s')
        elif key_list[count] == 'D':
            pyautogui.press('d')
        elif key_list[count] == 'F':
            pyautogui.press('f')
        count = count + 1
        if count == 18:
            break




