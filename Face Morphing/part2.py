# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:15:15 2020

@author: 90553
"""
import numpy as np  
import cv2
import dlib 
import os 
from pathlib import Path
path = str(Path(__file__).parent)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path + "/shape_predictor_68_face_landmarks.dat")

image = cv2.imread(path + "/deniro.jpg")
image_2 = cv2.imread(path + "/aydemirakbas.png")
center_x = np.asarray((image.shape[1]-1) / 2)
center_y = np.asarray((image.shape[0]-1) / 2)
center_x = center_x.astype(int)
center_y = center_y.astype(int)

center_x_2 = np.asarray((image_2.shape[1]-1) / 2)
center_y_2 = np.asarray((image_2.shape[0]-1) / 2)
center_x_2 = center_x_2.astype(int)
center_y_2 = center_y_2.astype(int)

subdiv = cv2.Subdiv2D((0,0,image.shape[0],image.shape[1]))
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
rectangles = detector(gray)
points = predictor(gray, rectangles[0])
for i in range(68):
    subdiv.insert((points.part(i).x, points.part(i).y))

subdiv.insert((0,0))
subdiv.insert((0,image.shape[1]-1))
subdiv.insert((image.shape[0]-1, 0))
subdiv.insert((image.shape[0]-1, image.shape[1]-1))
subdiv.insert((0,center_x))
subdiv.insert((image.shape[0]-1, center_x))
subdiv.insert((center_y,0))
subdiv.insert((center_y,image.shape[1]-1))


img_1_triangles = subdiv.getTriangleList()


indexes_triangles = []
clone = image.copy()
clone_2 = image_2.copy()

img_1_triangles = np.array(img_1_triangles, dtype=np.int32)

landmarks_points = []
for n in range(0, 68):
    x = points.part(n).x
    y = points.part(n).y
    landmarks_points.append((x, y))

landmarks_points.append((0, 0))
landmarks_points.append((0, 199))
landmarks_points.append((0, image.shape[1]-1))
landmarks_points.append((199, 0))
landmarks_points.append((199, image.shape[1]-1))
landmarks_points.append((image.shape[0]-1, 0))
landmarks_points.append((image.shape[0]-1, 199))
landmarks_points.append((image.shape[0]-1, image.shape[1]-1))

landmarks_points = np.array(landmarks_points, np.int32)


for t in img_1_triangles:
    pt1 = (t[0],t[1])
    pt2 = (t[2],t[3])
    pt3 = (t[4],t[5])
    cv2.line(clone, pt1, pt2, (0, 255, 0), 1)
    cv2.line(clone, pt2, pt3, (0, 255, 0), 1)
    cv2.line(clone, pt1, pt3, (0, 255, 0), 1)
    index_pt1 = np.where((landmarks_points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)
    index_pt2 = np.where((landmarks_points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)
    index_pt3 = np.where((landmarks_points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)
    
    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        indexes_triangles.append(triangle)
  

gray_2 = cv2.cvtColor(image_2 , cv2.COLOR_BGR2GRAY)
rectangles_2 = detector(gray_2)
points_2 = predictor(gray_2, rectangles_2[0])
landmarks_points_2 =[]

for i in range(68):
    x = points_2.part(i).x
    y = points_2.part(i).y
    landmarks_points_2.append((x,y))

landmarks_points_2.append((0, 0))
landmarks_points_2.append((0, 199))
landmarks_points_2.append((0, image_2.shape[1]-1))
landmarks_points_2.append((199, 0))
landmarks_points_2.append((199, image_2.shape[1]-1))
landmarks_points_2.append((image_2.shape[0]-1, 0))
landmarks_points_2.append((image_2.shape[0]-1, 199))
landmarks_points_2.append((image_2.shape[0]-1, image_2.shape[1]-1))

temp = []
for triangle_index in indexes_triangles:
    pt1 = landmarks_points_2[triangle_index[0]]
    pt2 = landmarks_points_2[triangle_index[1]]
    pt3 = landmarks_points_2[triangle_index[2]]
    cv2.line(clone_2, pt1, pt2, (0, 255, 0), 1)
    cv2.line(clone_2, pt3, pt2, (0, 255, 0), 1)
    cv2.line(clone_2, pt1, pt3, (0, 255, 0), 1)
    nodes = [pt1,pt2,pt3]
    temp.append(nodes)



              
cv2.imwrite(path + "/deniro_triangle.jpg",clone)
cv2.imwrite(path + "/aydemirakbas_triangle.png",clone_2)




