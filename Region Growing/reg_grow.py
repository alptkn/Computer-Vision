import numpy as np 
import time 
import pyautogui
import cv2
from collections import defaultdict
from pathlib import Path 
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import exposure
import os 
from PIL import Image
import mpld3
import collections
from scipy import ndimage, misc
import argparse
d = Path(__file__).resolve().parents[1]



#This Function plots the 2-D slices of the 3-D image 
def show_slices(slices,p):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.savefig(str(d) + "/Project/Plot_" + str(i) + "_" + str(p))



#Get desired number neighbor pixels 
def get_neighbors(pt, checked, dims, p):
    neighbors = []

    if p == 4:
        if (pt[1] > 0) and not checked[pt[0], pt[1] - 1, pt[2]]:
            neighbors.append((pt[0], pt[1] - 1, pt[2]))
        if (pt[0] > 0) and not checked[pt[0] - 1, pt[1], pt[2]]:
            neighbors.append((pt[0] - 1, pt[1], pt[2]))

        if (pt[1] < dims[1] - 1) and not checked[pt[0], pt[1] + 1, pt[2]]:
            neighbors.append((pt[0], pt[1] + 1, pt[2]))
        if (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1], pt[2]]:
            neighbors.append((pt[0] + 1, pt[1], pt[2]))

    if p == 6:
        if (pt[0] > 0) and not checked[pt[0] - 1, pt[1], pt[2]]:
            neighbors.append((pt[0] - 1, pt[1], pt[2]))
        if (pt[1] > 0) and not checked[pt[0], pt[1] - 1, pt[2]]:
            neighbors.append((pt[0], pt[1] - 1, pt[2]))
        if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2] - 1]:
            neighbors.append((pt[0], pt[1], pt[2] - 1))

        if (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1], pt[2]]:
            neighbors.append((pt[0] + 1, pt[1], pt[2]))
        if (pt[1] < dims[1] - 1) and not checked[pt[0], pt[1] + 1, pt[2]]:
            neighbors.append((pt[0], pt[1] + 1, pt[2]))
        if (pt[2] < dims[2] - 1) and not checked[pt[0], pt[1], pt[2] + 1]:
            neighbors.append((pt[0], pt[1], pt[2] + 1))

    if p == 8:

        if (pt[1] > 0) and not checked[pt[0], pt[1] - 1, pt[2]]:
            neighbors.append((pt[0], pt[1] - 1, pt[2]))
        if (pt[0] > 0) and not checked[pt[0] - 1, pt[1], pt[2]]:
            neighbors.append((pt[0] - 1, pt[1], pt[2]))

        if (pt[1] < dims[1] - 1) and not checked[pt[0], pt[1] + 1, pt[2]]:
            neighbors.append((pt[0], pt[1] + 1, pt[2]))
        if (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1], pt[2]]:
            neighbors.append((pt[0] + 1, pt[1], pt[2]))

        if (pt[1] > 0) and (pt[0] > 0) and not checked[pt[0] - 1, pt[1] - 1, pt[2]]:
            neighbors.append((pt[0] - 1, pt[1] - 1, pt[2]))
        if (pt[1] < dims[1] - 1) and (pt[0] > 0) and not checked[pt[0] - 1, pt[1] + 1, pt[2]]:
            neighbors.append((pt[0] - 1, pt[1] + 1, pt[2]))
        if (pt[1] > 0) and (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1] - 1, pt[2]]:
            neighbors.append((pt[0] + 1, pt[1] - 1, pt[2]))
        if (pt[1] < dims[1] - 1) and (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1] + 1, pt[2]]:
            neighbors.append((pt[0] + 1, pt[1] + 1, pt[2]))


    if p == 26:
        if (pt[0] > 0) and not checked[pt[0] - 1, pt[1], pt[2]]:
            neighbors.append((pt[0] - 1, pt[1], pt[2]))
        if (pt[1] > 0) and not checked[pt[0], pt[1] - 1, pt[2]]:
            neighbors.append((pt[0], pt[1] - 1, pt[2]))
        if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2] - 1]:
            neighbors.append((pt[0], pt[1], pt[2] - 1))

        if (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1], pt[2]]:
            neighbors.append((pt[0] + 1, pt[1], pt[2]))
        if (pt[1] < dims[1] - 1) and not checked[pt[0], pt[1] + 1, pt[2]]:
            neighbors.append((pt[0], pt[1] + 1, pt[2]))
        if (pt[2] < dims[2] - 1) and not checked[pt[0], pt[1], pt[2] + 1]:
            neighbors.append((pt[0], pt[1], pt[2] + 1))

        if (pt[1] > 0) and (pt[0] > 0) and not checked[pt[0] - 1, pt[1] - 1, pt[2]]:
            neighbors.append((pt[0] - 1, pt[1] - 1, pt[2]))
        if (pt[1] < dims[1] - 1) and (pt[0] > 0) and not checked[pt[0] - 1, pt[1] + 1, pt[2]]:
            neighbors.append((pt[0] - 1, pt[1] + 1, pt[2]))
        if (pt[1] > 0) and (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1] - 1, pt[2]]:
            neighbors.append((pt[0] + 1, pt[1] - 1, pt[2]))
        if (pt[1] < dims[1] - 1) and (pt[0] < dims[0] - 1) and not checked[pt[0] + 1, pt[1] + 1, pt[2]]:
            neighbors.append((pt[0] + 1, pt[1] + 1, pt[2]))

        if (pt[1] > 0) and (pt[2] > 0) and not checked[pt[0], pt[1] - 1, pt[2]-1]:
            neighbors.append((pt[0], pt[1] - 1, pt[2]-1))
        if (pt[1] < dims[1] - 1) and (pt[2] > 0) and not checked[pt[0], pt[1] + 1, pt[2]-1]:
            neighbors.append((pt[0], pt[1] + 1, pt[2]-1))
        if (pt[1] > 0) and (pt[2] < dims[2] - 1) and not checked[pt[0], pt[1] - 1, pt[2]+1]:
            neighbors.append((pt[0], pt[1] - 1, pt[2]+1))
        if (pt[1] < dims[1] - 1) and (pt[2] < dims[2] - 1) and not checked[pt[0], pt[1] + 1, pt[2]+1]:
            neighbors.append((pt[0], pt[1] + 1, pt[2]+1))

        if (pt[0] > 0) and (pt[2] > 0) and not checked[pt[0]-1, pt[1], pt[2]-1]:
            neighbors.append((pt[0]-1, pt[1], pt[2]-1))
        if (pt[0] < dims[0] - 1) and (pt[2] > 0) and not checked[pt[0]+1, pt[1], pt[2]-1]:
            neighbors.append((pt[0]+1, pt[1], pt[2]-1))
        if (pt[0] > 0) and (pt[2] < dims[2] - 1) and not checked[pt[0]-1, pt[1], pt[2]+1]:
            neighbors.append((pt[0]-1, pt[1], pt[2]+1))
        if (pt[0] < dims[0] - 1) and (pt[2] < dims[2] - 1) and not checked[pt[0]+1, pt[1], pt[2]+1]:
            neighbors.append((pt[0]+1, pt[1], pt[2]+1))

        if (pt[0] < dims[0] - 1)and (pt[1] < dims[1] - 1) and (pt[2] < dims[2] - 1) and not checked[pt[0]+1, pt[1]+1, pt[2]+1]:
            neighbors.append((pt[0]+1, pt[1]+1, pt[2]+1))

        if (pt[0] < dims[0] - 1)and (pt[1] < dims[1] - 1) and (pt[2] >0) and not checked[pt[0]+1, pt[1]+1, pt[2]-1]:
            neighbors.append((pt[0]+1, pt[1]+1, pt[2]-1))

        if (pt[0] < dims[0] - 1)and (pt[1] >0) and (pt[2] < dims[2] - 1) and not checked[pt[0]+1, pt[1]-1, pt[2]+1]:
            neighbors.append((pt[0]+1, pt[1]-1, pt[2]+1))

        if (pt[0] > 0) and (pt[1] < dims[1] - 1) and (pt[2] < dims[2] - 1) and not checked[pt[0]-1, pt[1]+1, pt[2]+1]:
            neighbors.append((pt[0]-1, pt[1]+1, pt[2]+1))

        if (pt[0] < dims[0] - 1)and (pt[1] >0) and (pt[2] >0) and not checked[pt[0]+1, pt[1]-1, pt[2]-1]:
            neighbors.append((pt[0]+1, pt[1]-1, pt[2]-1))

        if (pt[0]>0)and (pt[1] >0) and (pt[2] < dims[2] - 1) and not checked[pt[0]-1, pt[1]-1, pt[2]+1]:
            neighbors.append((pt[0]-1, pt[1]-1, pt[2]+1))

        if (pt[0]>0)and (pt[1]  < dims[1] - 1) and (pt[2] >0) and not checked[pt[0]-1, pt[1]+1, pt[2]-1]:
            neighbors.append((pt[0]-1, pt[1]+1, pt[2]-1))

        if (pt[0]>0)and (pt[1]  > 0) and (pt[2] > 0) and not checked[pt[0]-1, pt[1]-1, pt[2]-1]:
            neighbors.append((pt[0]-1, pt[1]-1, pt[2]-1))




    return neighbors


def grow_region(img, seeds, t, p):
    
    #This array indicates the segmented are in the image 
    seg_volume = np.zeros(img.shape, dtype=np.bool)
    checked = np.zeros_like(seg_volume)
    #this aray contains seeds points 
    candidates = get_neighbors(seeds[0], checked, img.shape, p)
    
    for i,seed in enumerate(seeds):
        if i == 0:
            continue
        
        seg_volume[seed] = True
        checked[seed] = True
        candidates += get_neighbors(seed, checked, img.shape, p)
    
    while len(candidates) > 0:
            #print(".........................")
            pt = candidates.pop()
    
           
            if checked[pt]: continue
    
            checked[pt] = True
    
            # Handle borders.
            imin = max(pt[0]-t, 0)
            imax = min(pt[0]+t, img.shape[0]-1)
            jmin = max(pt[1]-t, 0)
            jmax = min(pt[1]+t, img.shape[1]-1)
            kmin = max(pt[2]-t, 0)
            kmax = min(pt[2]+t, img.shape[2]-1)
            
            #taking the mean of the neighboring pixels 
            if img[pt] >= img[imin:imax+1, jmin:jmax+1, kmin:kmax+1].mean():
               
                seg_volume[pt] = True
                #adding the point as a seed point into candidate list 
                candidates += get_neighbors(pt, checked, img.shape, p)

    return seg_volume


#dice score calculation 
def dice_score(img_1, img_2):
    confusion_matrix_arrs = {}
    groundtruth = img_1
    predicted = img_2
    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)
    confusion_matrix_arrs['tp'] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs['fp'] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs['fn'] = np.logical_and(groundtruth, predicted_inverse)
    
    
    
    
    counter_tp = 0
    counter_tn = 0
    counter_fp = 0
    counter_fn = 0
    for i in range(0,200):
        for j in range(0,100):
            temp = collections.Counter(confusion_matrix_arrs['tp'][i][j])
            counter_tp += temp[1]
            temp = collections.Counter(confusion_matrix_arrs['tn'][i][j])
            counter_tn += temp[1]
            temp = collections.Counter(confusion_matrix_arrs['fp'][i][j])
            counter_fp += temp[1]
            temp = collections.Counter(confusion_matrix_arrs['fn'][i][j])
            counter_fn += temp[1]
    
    dice = 2*counter_tp /(2*counter_tp + counter_fn + counter_fp)
    print(f'Dice score: {dice}')    



parser = argparse.ArgumentParser()
parser.add_argument('--num_seed', type=int, default=15, help='Number of seed')
parser.add_argument('--num_neigh', type=int, default=8, help='Number of neighbors')
opt = parser.parse_args()






img = nib.load(str(d) + "/Project/V.nii")
img_data = img.get_fdata()

ground_truth = nib.load(str(d) + "/Project/V_seg.nii")
img_ground_truth = ground_truth.get_fdata()

noise = np.load(str(d) + "/Project/noise.npy")
img_data = img_data + 0.1 * noise
seeds = []
#randomly generate seed points 



seeds.append((81,40,43))
seeds.append((80,32,44))
seeds.append((19,18,45))
seeds.append((104,77,46))
seeds.append((131,25,47))
seeds.append((82,41,48))
seeds.append((151,3,49))
seeds.append((53,27,50))
seeds.append((168,69,51))
seeds.append((121,20,52))
seeds.append((104,23,53))
seeds.append((82,38,54))
seeds.append((81,40,55))


"""
seeds.append((82,25,48))
seeds.append((81,40,43))
seeds.append((80,32,44))
seeds.append((19,18,45))
seeds.append((104,77,46))
"""

#Median filtering to denoise the image 
median_img = ndimage.median_filter(img_data, size=7)
binaryImg = grow_region(median_img,seeds,3,opt.num_neigh)





slice_0 = binaryImg[100, :, :]
slice_1 = binaryImg[:, 50, :]
slice_2 = binaryImg[:, :, 50]
show_slices([slice_0, slice_1, slice_2],opt.num_neigh)
plt.suptitle("Center slices for EPI image")  


dice_score(img_ground_truth, binaryImg)




new_img = np.zeros((200,100,100))
for i in range(0,200):
    for j in range(0,100):
        for k in range(0,100):
            if(binaryImg[i][j][k] == True):
                new_img[i][j][k] = 1



new_image = nib.Nifti1Image(new_img, np.eye(4))
new_image.set_data_dtype(np.int8)

nib.save(new_image, str(d) + '/Project/my_arr.nii.gz')              




