# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def overlap(a,b,area_overlap):
    
    
    
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
    
    
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / area_combined
    
    
        
    return iou
                        
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
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    
    
    iou = area_overlap / area_a
    return iou

def inside(a,b):
    
    
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
    area_b = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
    
    
    val = area_overlap / area_b
    return val



def merge_bbox(bb_nms):
    minc = min(bb_nms[:,0])
    minr = min(bb_nms[:,1])
    maxc = max(bb_nms[:,2])
    maxr = max(bb_nms[:,3])
            
    return [minc,minr,maxc,maxr]
            
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

def non_max_suppression2(boundingbox):
    
    
    
    iou_threshold = 0.5
    inside_threshold = 0.3
    
    
    loop = True
    
    while loop:
        bb_nms = []
        
        bb_luas = (boundingbox[:,2] - boundingbox[:,0]) * (boundingbox[:,3] - boundingbox[:,1])
    
        sort = np.argsort(-bb_luas)
        
        bb_luas = bb_luas[sort]
        boundingbox = boundingbox[sort]
        i = 0
        loop = False
        while i < len(boundingbox) and len(boundingbox) > 0:
            highest_bbox = boundingbox[i]
            
            iou_val = iou(highest_bbox,boundingbox)
            iou_val[iou_val > iou_threshold] = True
            iou_val[iou_val < iou_threshold] = False
            
    #        iou_val = iou_val.astype(bool)
            
            
            inside_val = inside(highest_bbox,boundingbox) 
            inside_val[inside_val > inside_threshold] = True
            inside_val[inside_val < inside_threshold] = False
            
    #        inside_val = inside_val.astype(bool)
            
            cek_bbox = np.logical_or(iou_val,inside_val)
    #        
            index = np.where(cek_bbox == True )
            
            if len(index[0]) > 1:
                bb_nms.append(merge_bbox(boundingbox[index]))
                boundingbox = np.delete(boundingbox,index, axis = 0)
                bb_luas = np.delete(bb_luas,index)
                
                i = 0
                loop = True
                
            else :
                i = i + 1
        
        if len(boundingbox) > 0 and len(bb_nms) > 0:
            boundingbox = np.concatenate((boundingbox,bb_nms), axis = 0)
        elif len(boundingbox) == 0 and len(bb_nms) > 0:
            boundingbox = np.array(bb_nms)
        
        
        
        
        
        
    return boundingbox
    
    
    
    
    
    
    
    