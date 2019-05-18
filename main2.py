# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:25:07 2019

@author: user
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2

from skimage import measure, color
from skimage.segmentation import  felzenszwalb,mark_boundaries, slic
from selective_search import selective_search
from cnn_model import cnn


def show_bbox(image,list_bbox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.imshow(image)

    for bbox in list_bbox:
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    ax.set_axis_off()
    plt.tight_layout()
    
    
    plt.show()


image = cv2.imread('coba_img/crop001522.png')


'''  percobaan '''
#image = plt.imread('percobaan2.jpg')

segmen1 = felzenszwalb(image, scale = 2, sigma = 0.8, min_size = 200)
segmen = slic(image, n_segments = 200, compactness = 10, sigma = 1)

stack = np.hstack((color.label2rgb(segmen1),color.label2rgb(segmen)))

#plt.imshow(stack)


''' percobaan '''


''' main '''

selective = selective_search(image)

#list_bbox = np.array(selective.boundingbox)

#list_img = selective.image_crop

#hasil, index = cnn(list_img)
#show_bbox(image,list_bbox[index])




