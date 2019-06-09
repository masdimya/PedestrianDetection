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
from bbox_operator import iou, non_max_suppression

import glob


def show_bbox(image,list_bbox):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    font_style = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    print(list_bbox)
    bb_color = (255, 0, 0)
    font_color = (255, 0, 0)
    
    for bbox in list_bbox:
        minr, minc, maxr, maxc = bbox
        w = maxc - minc
        h = maxr - minr 
        
        cv2.rectangle(image, (minc, minr), (minc+w, minr+h),bb_color , 3)
        
        
#        cv2.putText(image, 'human', (minc, minr), font_style, font_scale, 
#                    font_color, lineType=cv2.LINE_AA)
        
    
    plt.imshow(image)
    

    


image = cv2.imread('coba_img/00000110a.png')

#bb = np.load('good.npy')

#show_bbox(image,bb)

#    
selective = selective_search(image)
#print("finish process ..... ")



''' main '''
    

list_bbox = np.array(selective.boundingbox)

list_img = selective.image_crop

hasil, index, prob = cnn(list_img)

#bbox = np.load('samplebbox.npy')
#prob = npa.load('samplebprob.npy')




arr = non_max_suppression(list_bbox[index],prob)

show_bbox(image,arr)




