import numpy as np 
import cv2
import moviepy.editor as mpy
import pickle 
from pathlib import Path





def cat_frames(path, background):
    hist_r = np.zeros((256))
    hist_g = np.zeros((256))
    hist_b = np.zeros((256))
    min_list = np.zeros((180,3))
    for i in range(180):
        new_frame = background.copy()
        name = 'cat_' + str(i) + '.png'
        image = cv2.imread(path + str(name))
        image = np.fliplr(image)
        image_g_channel = image [ : , : , 1 ]#Green ch annel o f the image
        image_r_channel = image [ : , : , 2 ]#Red ch annel o f the image
        foreground = np.logical_or(image_g_channel < 180, image_r_channel > 150)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        nonzero_cat_values = image[nonzero_x, nonzero_y, :]
        #shifting cat image to right side of background image 
        x = 925 - np.amax(nonzero_y)
        nonzero_y += x
        source = new_frame[nonzero_x, nonzero_y,:]
        
        #histograms calculation 
        hist_blue,_ = np.histogram(source[:,0].flatten(), 256,[0,256])
        hist_green,_ = np.histogram(source[:,1].flatten(), 256,[0,256])
        hist_red,_ = np.histogram(source[:,2].flatten(), 256,[0,256])
        min_list[i][0] = np.amin(source[:,0])
        min_list[i][1] = np.amin(source[:,1])
        min_list[i][2] = np.amin(source[:,2])
        hist_r = hist_r + hist_red
        hist_g = hist_g + hist_green
        hist_b = hist_b + hist_blue
    #return average of the hitograms 
    return hist_r/180, hist_g/180, hist_b/180, min_list 

def hist_match(source,cdf_s, cdf_t,n,m,M,R,C,red,green):
    K = np.zeros((R,C))
    for i in range(m,M):
        while n < 255 and cdf_s[i] < 1 and cdf_t[n] < cdf_s[i]:
            n = n + 1 
        x,y = np.nonzero(source == i)
        values = source[x,y]
        values = 1
        K[x,y] = K[x,y] + n*values
    return K



path = str(Path(__file__).parent)
path_cat = path + str('/cat/')
background = cv2.imread("Malibu.jpg")
background_height = background.shape[0]
backgroud_width = background.shape[1]
ratio = 360/background_height
background = cv2.resize(background, (int(backgroud_width * ratio), 360))
images_list = []
hist_red, hist_green, hist_blue, min_list = cat_frames(path_cat, background)
min_list = min_list.astype(int)

cdf_red = hist_red.cumsum()
cdf_red = cdf_red / cdf_red[-1]
cdf_green = hist_green.cumsum()
cdf_green = cdf_green / cdf_green[-1]
cdf_blue = hist_blue.cumsum()
cdf_blue = cdf_blue / cdf_blue[-1]
for i in range(180):
        new_frame = background.copy()
        name = 'cat_' + str(i) + '.png'
        image = cv2.imread(path_cat + str(name))
        image_r = np.fliplr(image)
        image_g_channel = image [ : , : , 1 ]#Green ch annel o f the image
        image_r_channel = image [ : , : , 2 ]#Red ch annel o f the image
        image_g_channel_r = image_r [ : , : , 1 ]#Green ch annel o f the image
        image_r_channel_r = image_r [ : , : , 2 ]#Red ch annel o f the image
        foreground = np.logical_or(image_g_channel < 180, image_r_channel > 150)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        nonzero_cat_values = image[nonzero_x, nonzero_y, :]
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
        foreground_2 = np.logical_or(image_g_channel_r < 180, image_r_channel_r > 150)
        nonzero_x, nonzero_y = np.nonzero(foreground_2)
        nonzero_cat_values = image_r[nonzero_x, nonzero_y, :]
        hist_cat_blue,_ = np.histogram(nonzero_cat_values[:,0].flatten(), 256, [0,256])
        hist_cat_green,_ = np.histogram(nonzero_cat_values[:,1].flatten(), 256, [0,256])
        hist_cat_red,_ = np.histogram(nonzero_cat_values[:,2].flatten(),256, [0,256])
        cdf_cat_red = hist_cat_red.cumsum()
        cdf_cat_red = cdf_cat_red / cdf_cat_red[-1]
        cdf_cat_green = hist_cat_green.cumsum()
        cdf_cat_green = cdf_cat_green / cdf_cat_green[-1]
        cdf_cat_blue = hist_cat_blue.cumsum()
        cdf_cat_blue = cdf_cat_blue / cdf_cat_blue[-1]
        source = image_r[:,:,0]
        R,C = source.shape
        blue = hist_match(image_r[:,:,2], cdf_cat_red, cdf_red, min_list[i][2], np.amin(image_r[:,:,2]), np.amax(image_r[:,:,2])
                         , R,C, image_r_channel_r, image_g_channel_r)
        source = image_r[:,:,1]
        R,C = source.shape
        green = hist_match(image_r[:,:,1], cdf_cat_green, cdf_green, min_list[i][1], np.amin(image_r[:,:,1]), np.amax(image_r[:,:,1])
                         , R,C, image_r_channel_r, image_g_channel_r)
        source = image_r[:,:,2]
        R,C = source.shape
        red = hist_match(image_r[:,:,0], cdf_cat_blue, cdf_blue, min_list[i][0], np.amin(image_r[:,:,0]), np.amax(image_r[:,:,0])
                         , R,C, image_r_channel_r, image_g_channel_r)

        z = np.zeros((360,640,3))
        z[:,:,0] = blue
        z[:,:,1] = green
        z[:,:,2] = red 
        cat_value = z[nonzero_x, nonzero_y]
        x = 925 - np.amax(nonzero_y)
        nonzero_y += x
        new_frame[nonzero_x, nonzero_y] = cat_value
        
        images_list.append(new_frame)
    

clip = mpy.ImageSequenceClip(images_list, fps=25)
audio = mpy.AudioFileClip(path + str('/selfcontrol_part.wav')).set_duration(clip.duration)
clip_new = clip.set_audio(audio)
clip_new.write_videofile(path + str('/part4_video.mp4'), codec='libx264')

