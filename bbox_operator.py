# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def overlap(a,b,area_overlap):
    
    list_iou = []
    
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
    
    
    
    
    for i in range(len(area_overlap)):
        if (area_overlap[i] < area_a) and (area_overlap[i] < area_b[i])  :
            
            area_combined = area_a + area_b[i] - area_overlap[i]
            iou = area_overlap[i] / area_combined
        
        elif (area_overlap[i] >= area_a ) :
            iou = area_overlap[i] / area_b[i]
        
        elif (area_overlap[i] >= area_b[i] ) :
            iou = area_overlap[i] / area_a
        
        list_iou.append(iou)
        
    return np.array(list_iou)
                        
def iou(a,b):
    
    
    x1 = np.maximum(a[1], b[:, 1])
    y1 = np.maximum(a[0], b[:, 0])
    x2 = np.minimum(a[3], b[:, 3])
    y2 = np.minimum(a[2], b[:, 2])

    # AREAS OF OVERLAP - Area where the boxes intersect
    height  = y2-y1
    width = x2 - x1
    
    height[height<0] = 0
    width[width<0] = 0
    
    area_overlap = width * height
    
    
    iou = overlap(a,b,area_overlap)
    return iou

def merge_bbox(bb_nms):
    
    fix_bb = []
    
    for i in bb_nms:
        
        if len(i) > 1 :
            print(i)
            minc = min(i[:,0])
            minr = min(i[:,1])
            maxc = max(i[:,2])
            maxr = max(i[:,3])
            
            fix_bb.append([minc,minr,maxc,maxr])
            
#            print(minc,minr,maxc,maxr)
        
        
    
    
    return fix_bb
    

def non_max_suppression(boundingbox,prob):
    
    bb_nms = []
    while len(boundingbox) > 0:
        
        sort = np.argsort(-prob)
    
        boundingbox = boundingbox[sort]
        prob = prob[sort]
        
        maxprob = boundingbox[0]
        
        
        iou_val = iou(maxprob,boundingbox[1:])
        
        print(iou_val)
        
        non_max_thres = 0
        
        index = np.where(iou_val > non_max_thres)
        
        index = index[0] + 1
        
        index = np.append(index,0)
        
#        index = np.append(index,0)
        
        bb_nms.append(boundingbox[index])
        
        boundingbox = np.delete(boundingbox,index, axis = 0)
        prob = np.delete(prob,index, axis = 0)
        
        
        
        
        
#    return bb_nms[0]
    return merge_bbox(bb_nms)
    
    
    
    
    
    
    
    