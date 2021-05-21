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
for i in range(180):
    name = 'cat_' + str(i) + '.png'
    image = cv2.imread(path_cat + str(name))
    image_g_channel = image [ : , : , 1 ]#Green ch annel o f the image
    image_r_channel = image [ : , : , 2 ]#Red ch annel o f the image
    foreground = np.logical_or(image_g_channel < 180, image_r_channel > 150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x, nonzero_y, :]
    new_frame = background.copy()
    new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
    images_list.append(new_frame)

clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip(path + str('/selfcontrol_part.wav')).set_duration(clip.duration)
clip_new = clip.set_audio(audio)
clip_new.write_videofile(path + str('/part1_video.mp4'), codec='libx264')
                         


