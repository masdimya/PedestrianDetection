# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 08:09:18 2019

@author: ASUS
"""
import os
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_bbox(image,list_bbox,nama):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    bb_color = (255, 0, 0)
    
    for bbox in list_bbox:
        minr, minc, maxr, maxc = bbox
        w = maxc - minc
        h = maxr - minr 
        
        cv2.rectangle(image, (minc, minr), (minc+w, minr+h),bb_color , 3)
        
    plt.imshow(image)

def save_bbox(image,list_bbox,anotasi,path,nama,tipe):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    bb_color = (255, 0, 0)
    ano_color = (0,255,0)
    
    for bbox in list_bbox:
        minr, minc, maxr, maxc = bbox
        w = maxc - minc
        h = maxr - minr 
        
        cv2.rectangle(image, (minc, minr), (minc+w, minr+h),bb_color , 3)
    
    for bbox in anotasi:
        minr, minc, maxr, maxc = bbox
        w = maxc - minc
        h = maxr - minr 
        
        cv2.rectangle(image, (minc, minr), (minc+w, minr+h),ano_color , 3)
    plt.imsave(path+tipe+"/"+nama+"_"+tipe+".jpg",image)
    
    
    
def save_region(name,list_image):
    currentDT = datetime.datetime.now()
    path  = currentDT.strftime("%Y-%m-%d_%H.%M.%S")
    os.mkdir(name+'_'+path+'_crop')
         
    for i in range (len(list_image)):
        image = cv2.cvtColor(list_image[i], cv2.COLOR_BGR2RGB)
        plt.imsave(name+'_'+path+"_crop/segmen{}.jpg".format(i),image)
