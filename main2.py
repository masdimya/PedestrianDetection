# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:25:07 2019

@author: user
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import pandas as pd

from skimage import measure, color
from skimage.segmentation import  felzenszwalb,mark_boundaries, slic
from selective_search import selective_search
from cnn_model import cnn, cnn_keras

from bbox_operator import like_nms, akurasi, akurasi2, merge_bb, evaluasi
from saving import save_bbox,show_bbox,save_region



conf = 3
thres = 0.5
    
nama = 'person_246'
path = 'hasil/percobaan2/'


#image = cv2.imread('image/' +nama+'.png')


''' main '''

print('hgdhgd')
    
#selective = selective_search(image)


#for new image
#list_bbox = np.array(selective.boundingbox)
#list_img = selective.image_crop
#anotasi=[[0,0,0,0]]

#for test image
#list_bbox = np.load('bbox/{}_bbox.npy'.format(nama))
#list_img = np.load('list_img/{}_list.npy'.format(nama))
#read_anotasi = pd.read_csv('INRIAPerson/Test/annotations_new/{}.csv'.format(nama))
#read_anotasi = read_anotasi.values
#
#anotasi = np.zeros((read_anotasi.shape),dtype = int)
#anotasi[:,0]=read_anotasi[:,1]
#anotasi[:,1]=read_anotasi[:,0]
#anotasi[:,2]=read_anotasi[:,3]
#anotasi[:,3]=read_anotasi[:,2]
#
#
#index, prob = cnn_keras(list_img)
##hasil, index, prob = cnn(list_img)
#if len(index) > 0:
#    hasilcnn = np.unique(list_bbox[index],axis=0)
#    luas_image = image.shape[0]*image.shape[1]
#    bb_luas =  ((hasilcnn[:,2] - hasilcnn[:,0]) * (hasilcnn[:,3] - hasilcnn[:,1]))/luas_image
#    
#    index = np.where(bb_luas>0.6)
#    if(len(index[0])> 0):
#        boundingbox = np.delete(hasilcnn,index, axis = 0)
#    else :
#        boundingbox = hasilcnn
#
#    merge,banyak,notmerge = merge_bb(image,boundingbox, conf , thres)
#    
#    TP,FP,FN = akurasi(merge,anotasi)
#    recall,precission,f1score = evaluasi(TP,FP,FN)
#  
#    save_bbox(image,list_bbox,anotasi,path,nama,'selective')
#    save_bbox(image,boundingbox,anotasi,path,nama,'cnn')
#    save_bbox(image,merge,anotasi,path,nama,'merge')
#    save_bbox(image,notmerge,anotasi,path,nama,'notmerge')
#
#
#else :
#    TP,FP,FN = akurasi([],anotasi)
#    recall,precission,f1score = evaluasi(TP,FP,FN)
#      
#    save_bbox(image,list_bbox,anotasi,path,nama,'selective')
#    save_bbox(image,[[0,0,0,0]],anotasi,path,nama,'cnn')
#    save_bbox(image,[[0,0,0,0]],anotasi,path,nama,'merge')
#    save_bbox(image,[[0,0,0,0]],anotasi,path,nama,'notmerge')
#    
#








