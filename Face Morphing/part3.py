# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:17:09 2020

@author: 90553
"""
import numpy as np
import cv2
import dlib
import pickle 
#import moviepy.editor as mpy 
import os 
from pathlib import Path
path = str(Path(__file__).parent)


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def make_homogeneous(triangle):
    homogeneous = np.array([triangle[::2],triangle[1::2],[1,1,1]])
    return homogeneous

def calc_transform(triangle1, triangle2):
    
    source = make_homogeneous(triangle1).T
    target = triangle2
    Mtx = np.array([np.concatenate((source[0],np.zeros(3))),\
                    np.concatenate((np.zeros(3),source[0])),\
                    np.concatenate((source[1],np.zeros(3))),\
                    np.concatenate((np.zeros(3),source[1])),\
                    np.concatenate((source[2],np.zeros(3))),\
                    np.concatenate((np.zeros(3),source[2]))])
    
    coefs = np.matmul(np.linalg.pinv(Mtx),target)
    Transform = np.array([coefs[:3],coefs[3:],[0,0,1]])
    return Transform 

def vectorised_Bilinear(coordinates,target_img,size):
    coordinates[0] = np.clip(coordinates[0],0,size[0]-1)
    coordinates[1] = np.clip(coordinates[1],0,size[1]-1)
    lower = np.floor(coordinates).astype(np.uint32)
    upper = np.ceil(coordinates).astype(np.uint32)
    
    error = coordinates - lower 
    resindual = 1 - error 

    
    top_left = np.multiply(np.multiply(resindual[0],resindual[1]).reshape(coordinates.shape[1],1),target_img[lower[0],lower[1],:])
    top_right = np.multiply(np.multiply(resindual[0],error[1]).reshape(coordinates.shape[1],1),target_img[lower[0],upper[1],:])
    bot_left = np.multiply(np.multiply(error[0],resindual[1]).reshape(coordinates.shape[1],1),target_img[upper[0],lower[1],:])
    bot_right = np.multiply(np.multiply(error[0],error[1]).reshape(coordinates.shape[1],1),target_img[upper[0],upper[1],:])
    
    z = np.uint8(np.round(top_left + top_right + bot_left + bot_right))
    return z

def image_morph(image1,image2,triangle1,triangle2,transforms, t):
    inter_image_1 = np.zeros(image1.shape).astype(np.uint8)
    inter_image_2 = np.zeros(image2.shape).astype(np.uint8)
    
    for i in range(len(transforms)):
        homo_inter_tri = (1-t)*make_homogeneous(triangle1[i]) + t*make_homogeneous(triangle2[i])
        polygon_mask = np.zeros(image1.shape[:2], dtype=np.uint8)
        cv2.fillPoly(polygon_mask,[np.int32(np.round(homo_inter_tri[1::-1,:].T))], color=255)
        seg = np.where(polygon_mask == 255)
        mask_points = np.vstack((seg[0],seg[1],np.ones(len(seg[0]))))
        inter_tri = homo_inter_tri[:2].flatten(order="F")
        inter_to_img1 = calc_transform(inter_tri, triangle1[i])
        inter_to_img2 = calc_transform(inter_tri, triangle2[i])
        mapped_to_img1 = np.matmul(inter_to_img1, mask_points)[:-1]
        mapped_to_img2 = np.matmul(inter_to_img2, mask_points)[:-1]
        inter_image_1[seg[0],seg[1],:] = vectorised_Bilinear(mapped_to_img1, image1, inter_image_1.shape)
        inter_image_2[seg[0],seg[1],:] = vectorised_Bilinear(mapped_to_img2, image2, inter_image_2.shape)
    result = (1-t)*inter_image_1 + t*inter_image_2
    return result.astype(np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path + "/shape_predictor_68_face_landmarks.dat")

image = cv2.imread(path + "/deniro.jpg")
image_2 = cv2.imread(path + "/panda.jpg")


gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

rectangles = detector(gray)
landmarks = predictor(gray, rectangles[0])
landmarks_2 = np.load(path + "/panda_landmarks.npy")

landmark_points_1 = []
for n in range(0,68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmark_points_1.append((x,y))

landmark_points_1.append((0, 0))
landmark_points_1.append((0, 199))
landmark_points_1.append((0, image.shape[1]-1))
landmark_points_1.append((199, 0))
landmark_points_1.append((199, image.shape[1]-1))
landmark_points_1.append((image.shape[0]-1, 0))
landmark_points_1.append((image.shape[0]-1, 199))
landmark_points_1.append((image.shape[0]-1, image.shape[1]-1))
points_1 = np.array(landmark_points_1, dtype=np.int32)

subdiv = cv2.Subdiv2D((0,0,image.shape[0],image.shape[1]))
subdiv.insert(landmark_points_1)

subdiv.insert((0,0))
subdiv.insert((0,image.shape[1]-1))
subdiv.insert((image.shape[0]-1, 0))
subdiv.insert((image.shape[0]-1, image.shape[1]-1))
subdiv.insert((0,199))
subdiv.insert((image.shape[0]-1, 199))
subdiv.insert((199,0))
subdiv.insert((199,image.shape[1]-1))

img_1_triangles = subdiv.getTriangleList()

index_triangles = []
for t in img_1_triangles:
    pt1 = (t[0],t[1])
    pt2 = (t[2],t[3])
    pt3 = (t[4],t[5])
    
    index_pt1 = np.where((points_1==pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)
    
    index_pt2 = np.where((points_1==pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)
    
    index_pt3 = np.where((points_1==pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)
    
    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        index_triangles.append(triangle)

landmark_points_2 = []
for n in range(0,68):
    x = landmarks_2[n][0]
    y = landmarks_2[n][1]
    landmark_points_2.append((x,y))

landmark_points_2.append((0, 0))
landmark_points_2.append((0, 199))
landmark_points_2.append((0, image_2.shape[1]-1))
landmark_points_2.append((199, 0))
landmark_points_2.append((199, image_2.shape[1]-1))
landmark_points_2.append((image_2.shape[0]-1, 0))
landmark_points_2.append((image_2.shape[0]-1, 199))
landmark_points_2.append((image_2.shape[0]-1, image_2.shape[1]-1))

points_2 = np.array(landmark_points_2, dtype=np.int32)


clone = image.copy()
clone_2 = image_2.copy()

temp = []
temp_2 = []
for triangle_index in index_triangles:
    
    tr2_pt1 = landmark_points_2[triangle_index[0]]
    tr2_pt2 = landmark_points_2[triangle_index[1]]
    tr2_pt3 = landmark_points_2[triangle_index[2]]
    nodes = [tr2_pt1,tr2_pt2,tr2_pt3]
    temp_2.append(nodes)
    
    

triangle_img_1 = np.array(img_1_triangles, dtype=np.int32)
triangle_img_2 = np.array(temp_2, dtype=np.int32)

z = np.where(triangle_img_1 == -1200)
temp = np.delete(triangle_img_1, z[0], axis=0)
z = np.where(temp == 1200)
temp = np.delete(temp, z[0], axis=0)


img_1_triangles = temp.reshape(-1,6)
img_2_triangles = triangle_img_2.reshape(-1,6)


img_1_triangles = img_1_triangles[:,[1,0,3,2,5,4]]
img_2_triangles = img_2_triangles[:,[1,0,3,2,5,4]]
Transforms = np.zeros((len(img_1_triangles),3,3))
for i in range(len(img_1_triangles)):
    source = img_1_triangles[i]
    target = img_2_triangles[i]
    Transforms[i] = calc_transform(source,target)

morphs = []
for t in(np.arange(0, 1.0001, 0.02)):
    a = image_morph(image, image_2, img_1_triangles, img_2_triangles, Transforms, t)[:,:,::-1]
    morphs.append(a)



with open(path + "/results", "wb") as f:
    pickle.dump(morphs, f)