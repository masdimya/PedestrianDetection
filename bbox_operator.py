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
    print('union :',area_combined)
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
    
    
    
#    iou = area_overlap / area_a
    return overlap(a,b,area_overlap)

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

def non_max_suppression2(image,boundingbox):
    
    luas_image = image.shape[0]*image.shape[1]
    bb_luas =  ((boundingbox[:,2] - boundingbox[:,0]) * (boundingbox[:,3] - boundingbox[:,1]))/luas_image
#    
#    index = np.where(bb_luas > 0.8)
#    boundingbox = np.delete(boundingbox,index, axis = 0)
#    
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
    
            inside_val = inside(highest_bbox,boundingbox) 
            inside_val[inside_val > inside_threshold] = True
            inside_val[inside_val < inside_threshold] = False

            cek_bbox = np.logical_or(iou_val,inside_val)       
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

def nms_by_prob(image,boundingbox,prob):
    
    luas_image = image.shape[0]*image.shape[1]
    bb_luas =  ((boundingbox[:,2] - boundingbox[:,0]) * (boundingbox[:,3] - boundingbox[:,1]))/luas_image
    
    bb_nms=[]
    
    iou_threshold = 0.5

    while len(boundingbox) > 0:
        sort = np.argsort(-prob)
        boundingbox = boundingbox[sort]
        prob = prob[sort]
        
        highest_bbox = boundingbox[0]
        
        iou_val = iou(highest_bbox,boundingbox)
        iou_val[iou_val > iou_threshold] = True
        iou_val[iou_val < iou_threshold] = False
   
        index = np.where(iou_val == True )
        
        if len(index[0]) > 1:
            boundingbox = np.delete(boundingbox,index, axis = 0)
            prob = np.delete(prob,index, axis = 0)
            bb_nms.append(highest_bbox)
        
        bb_nms.append(highest_bbox)
        boundingbox = np.delete(boundingbox,[0], axis = 0)
        prob = np.delete(prob,[0], axis = 0)
    return bb_nms

def merge_bb(image,boundingbox,conf,thres):
    
    iou_threshold = thres
    
    overlap_bb = []
    save_overlap = []
    
    
    for i in range(len(boundingbox)):
        iou_val = iou(boundingbox[i],boundingbox)
        index = np.where(iou_val > iou_threshold )
        overlap_bb.append(len(index[0])-1)
        save_overlap.append(boundingbox[index])
        
    
    overlap_bb = np.array(overlap_bb)
    
    
    if len(overlap_bb) > 0 and overlap_bb.max() > 0 :
          index_bb = np.where(overlap_bb>conf)
          high_overlap = boundingbox[index_bb]
          boundingbox = boundingbox[index_bb]
          luas_image = image.shape[0]*image.shape[1]
          bb_luas =  ((boundingbox[:,2] - boundingbox[:,0]) * (boundingbox[:,3] - boundingbox[:,1]))/luas_image

          loop = True
    
          while loop:
              print('adsad')
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
                  
                  morethres=np.where(iou_val > iou_threshold )
                  underthres = np.where((iou_val < iou_threshold) & (iou_val > 0)   )
                  zero = np.where(iou_val <= 0)
                  
                  
    
                  index = np.where(iou_val == True )
    
                  if len(morethres[0]) > 1 or len(underthres[0]) > 1 :
                      if len(morethres[0]) > 1 and len(underthres[0]) > 1:
                          bb_nms.append(merge_bbox(boundingbox[morethres]))
                          delindex = np.concatenate((morethres[0],underthres[0]))
                          
                          boundingbox = np.delete(boundingbox,delindex, axis = 0)
                          bb_luas = np.delete(bb_luas,delindex)
                          
                          i = 0
                          loop = True
                        
                        
                      elif len(morethres[0]) > 1 and len(underthres[0]) < 1:
                          bb_nms.append(merge_bbox(boundingbox[morethres]))
                          boundingbox = np.delete(boundingbox,morethres, axis = 0)
                          bb_luas = np.delete(bb_luas,morethres)
                          
                          i = 0
                          loop = True
                      
                      elif len(morethres[0]) < 1 and len(underthres[0]) > 1:
                          underthres[0] = underthres[0]
                          boundingbox = np.delete(boundingbox,underthres, axis = 0)
                          bb_luas = np.delete(bb_luas,underthres)
                          
                          i = 0
                          loop = True
                      else:
                          i = i + 1
                      
    
                  else :
                      i = i + 1
    
              if len(boundingbox) > 0 and len(bb_nms) > 0:
                  boundingbox = np.concatenate((boundingbox,bb_nms), axis = 0)
              elif len(boundingbox) == 0 and len(bb_nms) > 0:
                  boundingbox = np.array(bb_nms)
              return boundingbox,overlap_bb,high_overlap
    else:
      return [],0,[]

def akurasi(predict,target):
    threshold = 0.2
    TP = 0
    FP = 0
    FN = len(target)
    
    
    if len(target) > 0 and len(predict) > 0:
      for i in range(len(predict)):
        iou_val = iou(predict[i],target)
        index = np.where(iou_val >=threshold)

        if len(index[0]) > 0:
            TP =TP + 1
            FN = FN - 1
            target = np.delete(target,index,axis = 0)
        else:
            FP=FP+1
    else:
      FP = len(predict)
    
    
    return TP,FP,FN

def akurasi2(predict,target):
    TP = 0
    FP = 0
    FN = len(target)
    
    
    if len(target) == 0 and len(predict) == 0:
      TP = 1
    else:
      FP = len(predict)
    
    
    return TP,FP,FN

def evaluasi(TP,FP,FN):
  
  if TP > 0:
    recall = TP/(TP+FN)
    precission = TP/(TP+FP)
  else :
    recall = 0
    precission = 0
  
  if precission > 0 or recall > 0:
    f1_score = (2*precission*recall)/(precission+recall)
  else :
    f1_score = 0
    
  return recall,precission,f1_score


    
    
    