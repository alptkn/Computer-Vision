import numpy as np 
import moviepy.editor as mpy 
import os 
import cv2
from pathlib import Path

path = str(Path(__file__).parent)

path_cat = path + str('/cat/')
background = cv2.imread("Malibu.jpg")
background_height = background.shape[0]
backgroud_width = background.shape[1]
ratio = 360/background_height
background = cv2.resize(background, (int(backgroud_width * ratio), 360))
images_list = []
reverse_img_list = []

def create_frame(image, new_frame, mode):
    image_g_channel = image [ : , : , 1 ]#Green ch annel o f the image
    image_r_channel = image [ : , : , 0 ]#Red ch annel o f the image
    foreground = np.logical_or(image_g_channel < 180, image_r_channel > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x, nonzero_y, :]
    
    #shifting cat image to right side of background image 
    if mode == 'r':
        x = 925 - np.amax(nonzero_y)
        nonzero_y += x
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
    return new_frame

for i in range(180):
    name = 'cat_' + str(i) + '.png'
    image = cv2.imread(path_cat + str(name))
    image_r = np.fliplr(image)
    mode = 'n'
    new_frame = background.copy()
    new_frame = create_frame(image, new_frame, mode)
    mode = 'r'
    new_frame = create_frame(image_r, new_frame, mode)
    images_list.append(new_frame)
    
    
clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip(path + str('/selfcontrol_part.wav')).set_duration(clip.duration)
clip_new = clip.set_audio(audio)
clip_new.write_videofile(path + str('/part2_video.mp4'), codec='libx264')
