# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:07:22 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

#class similarity_neighbor():
#    def __init__(self,region1,region2,segmentation):
#        self.image_region1 = region1.segmen
#        self.image_region2 = region2.segmen
#        
#        self.region1bbox = region1.bbox
#        self.region2bbox = region2.bbox
#        
#        self.sizeImage = segmentation.shape[0]*segmentation.shape[1]
#        
#        self.sim_color = self.similarity_color(self.image_region1,self.image_region2)
#        self.sim_size = self.similarity_size(self.image_region1,self.image_region2,self.sizeImage)
#        self.sim_fill = self.similarity_fill(self.region1bbox,self.region2bbox,self.sizeImage)
#        
#        self.sim_total = self.sim_color + self.sim_size + self.sim_fill
#        
#        print(self.sim_total)
#
#    def quantization(self,image):
#        pixel_range = [x for x in range(32,257,32)]
#        bins_hist = np.arange(1,9)
#        
#        list_hist = []
#        for chanel in range(3):
#            image_chanel = image[:,:,chanel]
#
#            for iter_pixel in range(len(pixel_range)):
#                if iter_pixel == 0 :
#                    image_chanel[(image_chanel >= 0)& (image_chanel< pixel_range[iter_pixel])] = 1
#                else :
#                    image_chanel[(image_chanel>= pixel_range[iter_pixel-1])& (image_chanel< pixel_range[iter_pixel])] = iter_pixel + 1
#
#
#            hist, bins = np.histogram(image_chanel, bins_hist )
#
#
#            if chanel == 0 :
#                list_hist = hist
#            else :
#                list_hist = np.concatenate((list_hist,hist))
#
#        return list_hist
#
#    def similarity_color(self,image1,image2):
#
#        quan1 = self.quantization(image1)
#        quan2 = self.quantization(image2)
#
#        minimum = np.minimum(quan1,quan2)
#        sim_color = sum(minimum)
#
#        return sim_color
#
#    def similarity_size(self,image1,image2,sizeImage):
#        sizeImage1 = image1.shape[0] * image1.shape[1]
#        sizeImage2 = image2.shape[0] * image2.shape[1]
#
#        sim_size = 1 - ( (sizeImage1+sizeImage2) / sizeImage )
#        return sim_size
#    
#    def similarity_fill(self,bboxRegion1,bboxRegion2, sizeImage):
#        r1_Ymin = bboxRegion1[0]
#        r1_Xmin = bboxRegion1[1]
#        r1_Ymax = bboxRegion1[2]
#        r1_Xmax = bboxRegion1[3]
#
#        r2_Ymin = bboxRegion2[0]
#        r2_Xmin = bboxRegion2[1]
#        r2_Ymax = bboxRegion2[2]
#        r2_Xmax = bboxRegion2[3]
#
#
#        BBox_Xmin = min(r1_Xmin,r2_Xmin)
#        BBox_Ymin = min(r1_Ymin,r2_Ymin)
#        BBox_Xmax = max(r1_Xmax,r2_Xmax)
#        BBox_Ymax = max(r1_Ymax,r2_Ymax)
#
#
#
#        size_r1 = (r1_Xmax-r1_Xmin) * (r1_Ymax - r1_Ymin )
#        size_r2 = (r2_Xmax-r2_Xmin) * (r2_Ymax - r2_Ymin )
#        size_bbox = (BBox_Xmax - BBox_Xmin ) * (BBox_Ymax - BBox_Ymin)
#
#        total = 1- ((size_bbox - size_r1 - size_r2 )/sizeImage ) 
#
#        return total



def quantization(image):
#    pixel_range = [x for x in range(32,257,32)]
#    bins_hist = np.arange(1,9)
#   
#    list_hist = []
#    for chanel in range(3):
#        image_chanel = image[:,:,chanel]
#
#        for iter_pixel in range(len(pixel_range)):
#            if iter_pixel == 0 :
#                image_chanel[(image_chanel >= 0)& (image_chanel< pixel_range[iter_pixel])] = 1
#            else :
#                image_chanel[(image_chanel>= pixel_range[iter_pixel-1])& (image_chanel< pixel_range[iter_pixel])] = iter_pixel + 1
#
#
#        hist, bins = np.histogram(image_chanel, bins_hist )
#
#
#        if chanel == 0 :
#            list_hist = hist
#        else :
#            list_hist = np.concatenate((list_hist,hist))
    list_hist = np.empty(0, dtype = int)
    
    for chanel in range(3):
        image_chanel = image[:,:,chanel]
        hist, bins = np.histogram(image_chanel, bins = 8 )
        
        list_hist = np.append(list_hist,hist)        
    
    return list_hist

def similarity_color(image1,image2):

    quan1 = quantization(image1) 
    quan2 = quantization(image2) 
    

    minimum = np.minimum(quan1,quan2)
    sim_color = sum(minimum)
    

    return sim_color

def similarity_size(image1,image2,sizeImage):
    sizeImage1 = image1.shape[0] * image1.shape[1]
    sizeImage2 = image2.shape[0] * image2.shape[1]

    sim_size = 1 - ( (sizeImage1+sizeImage2) / sizeImage )
    return sim_size

def similarity_fill(bboxRegion1,bboxRegion2, sizeImage):
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



    size_r1 = (r1_Xmax-r1_Xmin) * (r1_Ymax - r1_Ymin )
    size_r2 = (r2_Xmax-r2_Xmin) * (r2_Ymax - r2_Ymin )
    size_bbox = (BBox_Xmax - BBox_Xmin ) * (BBox_Ymax - BBox_Ymin)

    total = 1- ((size_bbox - size_r1 - size_r2 )/sizeImage ) 

    return total

def similarity_neighbor(region1,region2,segmentation):
    image_region1 = region1.segmen
    image_region2 = region2.segmen
    
    region1bbox = region1.bbox
    region2bbox = region2.bbox
    
    sizeImage = segmentation.shape[0]*segmentation.shape[1]
    

    
    sim_color = similarity_color(image_region1,image_region2)
    sim_size = similarity_size(image_region1,image_region2,sizeImage)
    sim_fill = similarity_fill(region1bbox,region2bbox,sizeImage)
    
    sim_total = sim_color + sim_size + sim_fill
    

    
    return sim_total
