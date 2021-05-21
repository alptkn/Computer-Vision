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
import pickle 
d = Path(__file__).resolve().parents[1]

vectors = []

#Lucas Kanade Algorithm 
def Lucas_Kanade(image1, image2, points, w,tau):
    oldframe = image1
    I1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
    
    newframe = image2
    I2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
    
    I1 = I1/255
    I2 = I2/255
    global vectors
   
    
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image
    
    #Taking derivatives 
    Ix = convolve2d(I1, Gx, boundary='symm', mode='same')
    Iy = convolve2d(I1, Gy, boundary='symm', mode='same')
    It1 = convolve2d(I1, Gt1, boundary='symm', mode='same') + convolve2d(I2,Gt2,boundary='symm', mode='same')
    
    
    
    
    
    
    u = np.ones(I1.shape)
    v = np.ones(I1.shape)
    

    newFeature=np.zeros_like(points)
    
    #Least square method 
    for a,i in enumerate(points):

        x,y = i
    
       
        f_x = Ix[y - w:y + w + 1, x - w:x + w + 1].flatten()
        f_y = Iy[y - w:y + w + 1, x - w:x + w + 1].flatten()
        f_t = It1[y - w:y + w + 1, x - w:x + w + 1].flatten()
        A = np.vstack((f_x, f_y)).T
        b = np.reshape(f_t, (f_t.shape[0],1))
        
        eigen = np.min(np.abs(np.linalg.eigvals(np.matmul(A.T, A))))
        """
        print("Eigen")
        print(eigen)
        """
        #if eigen values are greater than treshold, update u and v
        nu = np.matmul(np.linalg.pinv(A),b)
        if eigen >= tau:
            u[y,x]=nu[0]
            v[y,x]=nu[1]
        
        newFeature[a]=[np.int32(x+u[y,x]),np.int32(y+v[y,x])]
        if len(points) == 1:
                vectors.append(nu)
       
     
    for i in range(len(points)):
        x_old, y_old = points[i]
        x_new, y_new = newFeature[i]
        dx = int(u[y_old,x_old])
        dy = int(v[y_old,x_old])
        newframe = cv2.arrowedLine(image2.copy(),(x_old, y_old),(x_old + dx*10, y_old + dy*10),(0,0,255), 2)
   
    return newframe, newFeature,u,v

#subtract background from iamge 
def diffImage(image_1, image_2):
    temp = cv2.absdiff(image_1, image_2)
    temp2 = cv2.absdiff(image_1, temp)
    return temp2

#This function track the points 
def track(frames, frames_2):
    track_frames = []
    frame0 = frames[0]
    frame00 = diffImage(frames[0], frames_2[0])
    track_points_1 = np.array([[210,183], [308, 183],[210,315], [308, 315] ])
    track_points_2 = np.array([[401,321]])
    #print(track_points_1)
    
    for i in range(1, len(frames)):
        #print("Turn_" + str(i) +"---------------------------")
        #print(track_points_1)
        frame1 = frames[i]
        frame01 = diffImage(frames[i], frames_2[i])
        frame_back, track_points_1,_,_ = Lucas_Kanade(frame0, frame1, track_points_1,1,0.001)
        #print("Other----------------------------------------")
        _, temp,u,v = Lucas_Kanade(frame00, frame01, track_points_2, 1,0.001)
        for i in range(len(temp)):
            x_new, y_new = temp[i]
            x_old, y_old = track_points_2[i]
            dx = int(u[y_old,x_old])
            dy = int(v[y_old,x_old])
            frame = cv2.arrowedLine(frame_back.copy(), (x_old, y_old), (x_old + dx*10, y_old + dy*10),(255,0,0), 2)
        track_frames.append(frame)
        track_points_2 = temp
        frame0 = frame1
        frame00 = frame01
        
        
    
    return track_frames








frames = []
frames_2 = []
        

biped_vid = mpy.VideoFileClip(str(d) + "/HW4/biped_2.avi")
biped_vid_2 = mpy.VideoFileClip(str(d) + "/HW4/biped_1.avi")
frame_cout_2 = biped_vid_2.reader.nframes
frame_count = biped_vid.reader.nframes
video_fps = biped_vid.fps 
video_fps_2 = biped_vid_2.fps 



new_frames = []

for i in range(frame_count):
    walker_frame = biped_vid.get_frame(i*1.0/video_fps)
    walker_frame_2 = biped_vid_2.get_frame(i*1.0/video_fps_2)
    frames.append(walker_frame)
    frames_2.append(walker_frame_2)






new_frames = track(frames, frames_2)

with open(str(d) + "/HW4/part2_vectors", "wb") as f:
    pickle.dump(vectors, f)

clip = mp.ImageSequenceClip(new_frames, fps=video_fps)
clip.write_videofile(str(d) + "/HW4/walker_2.mp4", codec='libx264')


