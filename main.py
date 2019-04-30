# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:12:48 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import measure, color
from skimage.segmentation import  felzenszwalb

from selective_search import Region,get_neighbor 
from similarity import similarity_neighbor


def mergebbox(bboxRegion1,bboxRegion2):
    r1_Ymin = bboxRegion1[0]
    r1_Xmin = bboxRegion1[1]
    r1_Ymax = bboxRegion1[2]
    r1_Xmax = bboxRegion1[3]

    r2_Ymin = bboxRegion2[0]
    r2_Xmin = bboxRegion2[1]
    r2_Ymax = bboxRegion2[2]
    r2_Xmax = bboxRegion2[3]


    BBox_Xmin = min(r1_Xmin,r2_Xmin)
    BBox_Ymin = min(r1_Ymin,r2_Ymin)
    BBox_Xmax = max(r1_Xmax,r2_Xmax)
    BBox_Ymax = max(r1_Ymax,r2_Ymax)
    
    return [BBox_Xmin,BBox_Ymin,BBox_Xmax,BBox_Ymax]

a = plt.imread('../ujicoba.jpg')

segmen = felzenszwalb(a, scale = 200, sigma=0.5, min_size = 50)
banyak = segmen.max()
banyak = banyak+1
segmen[segmen == 0 ] = banyak

sizeImage = a.shape[0]*a.shape[1]
list_region = []
list_neighbor_pair = []
list_similarity = []

for region in measure.regionprops(segmen):
    img_segmen = color.gray2rgb(region.image) * a[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]]
    
    r = Region(region.label,region.bbox,img_segmen)
    
    list_neighbor_pair = list_neighbor_pair + get_neighbor(segmen,r)
    
    list_region.append(r)


for i in range(len(list_neighbor_pair)):
    
    ri = list_region[ list_neighbor_pair[i][0] - 1 ] 
    rj = list_region[ list_neighbor_pair[i][1] - 1 ] 
    
    list_similarity.append( similarity_neighbor( ri,rj ,segmen ) )


list_similarity = np.array(list_similarity)
list_neighbor_pair = np.array(list_neighbor_pair)


 
max_index = np.argmax(list_similarity)

max_sim = list_similarity[max_index]
ri,rj = list_neighbor_pair[max_index][0],list_neighbor_pair[max_index][1]

neighbor_index = np.concatenate((
        np.where(list_neighbor_pair[:,0] == ri),
        np.where(list_neighbor_pair[:,0] == rj)),axis = 1 )


list_neighbor_pair = np.delete(list_neighbor_pair,neighbor_index[0],axis = 0)
list_similarity = np.delete(list_similarity,neighbor_index[0],axis = 0)

other_neigh_index = np.concatenate((
        np.where(list_neighbor_pair[:,1] == ri),
        np.where(list_neighbor_pair[:,1] == rj)),axis = 1 )

other_neigh = list_neighbor_pair[other_neigh_index[0]]


list_neighbor_pair = np.delete(list_neighbor_pair,other_neigh_index[0],axis = 0)
list_similarity = np.delete(list_similarity,other_neigh_index[0],axis = 0)





banyak = banyak + 1 

rt_label = banyak
rt_bbox =  mergebbox(list_region[ri-1].bbox,list_region[rj-1].bbox)
#segmen2 = np.copy(segmen)



segmen[segmen == ri ] = rt_label
segmen[segmen == rj ] = rt_label

rt_image = np.copy(segmen[rt_bbox[0]:rt_bbox[2],rt_bbox[1]:rt_bbox[3]])
rt_image[rt_image != rt_label ] = 0
rt_image[rt_image == rt_label ] = 1



rt_image = color.gray2rgb(rt_image) * a[rt_bbox[0]:rt_bbox[2],rt_bbox[1]:rt_bbox[3]] 

r = Region(rt_label,rt_bbox,rt_image)

list_region.append(r)

rt_neighbor = get_neighbor(segmen,r)

other_neigh = np.unique(other_neigh[:,0])
new_other_neigh = np.zeros((other_neigh.shape[0],2), dtype=int)
new_other_neigh[:,0] = other_neigh 
new_other_neigh[:,1] = rt_label  

new_neighbor = np.concatenate((rt_neighbor,new_other_neigh), axis = 0)
new_sim = []

for i in range(len(new_neighbor)):
    
    ri = list_region[ new_neighbor[i][0] - 1 ] 
    rj = list_region[ new_neighbor[i][1] - 1 ] 
    
    new_sim.append( similarity_neighbor( ri,rj ,segmen ) )


list_neighbor_pair = np.append(list_neighbor_pair,new_neighbor,axis = 0)
list_similarity = np.append(list_similarity,new_sim)    


#list_neighbor_pair = list_neighbor_pair + get_neighbor(segmen,r)



plt.imshow(rt_image)


    




    
