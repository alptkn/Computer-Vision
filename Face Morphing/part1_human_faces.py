# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:59:37 2020

@author: 90553
"""

import numpy as np  
import cv2
import os 
import dlib 
from pathlib import Path
path = str(Path(__file__).parent)


def rectangle_to_bb(rectangles):
    x = rectangles[0].left()
    y = rectangles[0].top()
    w = rectangles[0].right() - x
    h = rectangles[0].bottom() - y
    return (x,y,w,h)

def draw_rect(image, x,y,x1,y1,x2,y2,x3,y3):
    clone = image.copy()
    for n in range(y,y1+1):
        clone[n][x][0] = 0
        clone[n][x][1] = 255
        clone[n][x][2] = 0
    for n in range(y2,y3+1):
        clone[n][x2][0] = 0
        clone[n][x2][1] = 255
        clone[n][x2][2] = 0
    for n in range(x,x2+1):
        clone[y][n][0] = 0
        clone[y][n][1] = 255
        clone[y][n][2] = 0
    for n in range(x1,x3+1):
        clone[y1][n][0] = 0
        clone[y1][n][1] = 255
        clone[y1][n][2] = 0
    
    return clone

def mark_landmarks(image, points):
    clone = image.copy()
    for i in range(0,68):
        x = points.part(i).x
        y = points.part(i).y
        clone[y][x][0] = 0
        clone[y][x][1] = 255
        clone[y][x][2] = 0

    return clone 

    


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/90553/Desktop/Computer_Vision/HW2/shape_predictor_68_face_landmarks.dat")



for name in ["deniro", "aydemirakbas", "kimbodnia"]:
    extent = "png"
    if name == "deniro":
        extent = "jpg"
    
    image = cv2.imread(path + "/" + str(name) + "." + str(extent))
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])
    
    #The function that marks the facial points 
    #clone_2 = mark_landmarks(image, points)
    clone = image.copy()
    for n in range(0,68):
        x = points.part(n).x
        y = points.part(n).y
        
        cv2.circle(clone, (x,y), 3, (0,255,0), -1 )
    
    cv2.imwrite(path + "/" + str(name) + "_part1_." + str(extent), clone)

