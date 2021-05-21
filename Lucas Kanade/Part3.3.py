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
import scipy
import scipy.ndimage.filters
from scipy.signal import convolve

d = Path(__file__).resolve().parents[1]

vectors = list()

def gaussian_filter(filter_size):
    if filter_size == 1:
        return np.array([[1]])

    filter = np.float64(np.array([[1, 1]]))

    for i in range(filter_size - 2):
        filter = scipy.signal.convolve2d(filter, np.array([[1, 1]]))

    return filter / np.sum(filter)

def blur_im(im, filter_vec):
    blur_rows = scipy.ndimage.filters.convolve(im, filter_vec)
    blur_columns = scipy.ndimage.filters.convolve(blur_rows, filter_vec.T)
    return blur_columns

def reduce(im, filter_vec):
    blurred_im = blur_im(im, filter_vec)
    return blurred_im[::2, ::2]

def expand(im, filter_vec):
    im_shape = im.shape
    expanded_im = np.zeros((2 * im_shape[0], 2 * im_shape[1]))
    expanded_im[::2, ::2] = im
    return blur_im(expanded_im, 2 * filter_vec)



def build_gaussian_pyramid(im, max_levels, filter_size):
    pyr = []
    filter_vec = gaussian_filter(filter_size)
    next_level_im = im
    max_levels = min(max_levels, int(np.log(im.shape[0] // 16) / np.log(2)) + 1,
                     int(np.log(im.shape[1] // 16) / np.log(2)) + 1)

    for i in range(max_levels):
        pyr.append(next_level_im)
        next_level_im = reduce(np.copy(next_level_im), filter_vec)

    return pyr, filter_vec



#Lucas Kanade Algorithm 
def Lucas_Kanade(image1, image2,image_3, points, w):
    I1 = oldframe = image1
    #I1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
    
    I2 = newframe = image2
    #I2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
   
    global vectors
    I1 = I1/255
    I2 = I2/255
   
   
    
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
    tau = 0.1
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
        Ainv = np.linalg.pinv(np.matmul(A.T,A))
        nu = np.matmul(Ainv, np.matmul(A.T,b))
         #if eigen values are greater than treshold, update u and v
        if eigen >= tau:
            u[y,x]=nu[0]
            v[y,x]=nu[1]
            
        newFeature[a]=[np.int32(x+u[y,x]),np.int32(y+v[y,x])]
        vectors.append(nu)
    temp = image2.copy()
    for i in range(len(newFeature)):
        #x_new, y_new = newFeature[i] 
        x_old, y_old = points[i]
        x_new, y_new = newFeature[i]
        dx = int(u[y_old,x_old])
        dy = int(v[y_old,x_old])
        newframe = cv2.arrowedLine(image_3.copy(),(x_old*2, y_old*2),(x_old*2 + dx*10, y_old*2 + dy*10),(0,0,255), 2)
    
    return newframe, newFeature

#subtract background from iamge 
def diffImage(image_1, image_2):
    temp = cv2.absdiff(image_1, image_2)
    temp2 = cv2.absdiff(image_1, temp)
    return temp2


#This function track the points 
def track(frames, frames_2):
    track_frames = []
    frame0 = diffImage(frames[0], frames_2[0])
    gray0 = cv2.cvtColor(frame0.copy(), cv2.COLOR_BGR2GRAY)
    imgs, vec = build_gaussian_pyramid(gray0, 2, 3)
    frame0_reduce = imgs[1]
    track_points = np.array([[200,160]])
    #print(track_points)
    
    for i in range(1, len(frames)):
        #print(i)
        #print(track_points)
        frame1 = diffImage(frames[i], frames_2[i])
        gray1 = cv2.cvtColor(frame1.copy(), cv2.COLOR_BGR2GRAY)
        imgs, vec = build_gaussian_pyramid(gray1, 3, 3)
        frame1_reduce = imgs[1]
        temp,track_points = Lucas_Kanade(frame0_reduce, frame1_reduce,frames[i].copy(), track_points,1)
        frame0_reduce = frame1_reduce
        track_frames.append(temp)
        
        
    
    return track_frames








frames = []
frames_2 = []
        
biped_vid_2 = mpy.VideoFileClip(str(d) + "/HW4/biped_1.avi")
biped_vid = mpy.VideoFileClip(str(d) + "/HW4/biped_3.avi")
frame_count_2 = biped_vid_2.reader.nframes
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

clip = mp.ImageSequenceClip(new_frames, fps=video_fps)
clip.write_videofile(str(d) + "/HW4/walker3.mp4", codec='libx264')




with open(str(d) + "/HW4/part3_corrected_vectors", "wb") as f:
    pickle.dump(vectors, f)


