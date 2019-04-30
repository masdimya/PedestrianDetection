# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:10:38 2019

@author: user
"""

import numpy as np
from skimage.segmentation import find_boundaries
from skimage import measure, color
from skimage.segmentation import  felzenszwalb


from similarity import similarity_neighbor



class Region():
    def __init__(self,label,bbox,segmen_area):
        self.label=label
        self.bbox=bbox
        self.segmen = segmen_area
        
        
class selective_search():
    def __init__(self,image):
        
        self.segmen = felzenszwalb(image, scale = 200, sigma=0.5, min_size = 50)
    
        self.max_label = self.fix_label(self.segmen)
        
        self.sizeImage = image.shape[0]*image.shape[1]
        
        self.hierarchy = []
        self.img_hierarchy = []
        
        
        
        self.list_region , self.list_neighbor_pair = self.extract_region (self.segmen,image) 
        
        self.list_similarity = self.similarity_neigh_pair(self.list_neighbor_pair, 
                                                          self.list_region,
                                                          self.segmen)
        
        self.merge_region(image)
        
        
    
    def fix_label (self,segmen) :
        count_label = segmen.max()
        
        count_label = count_label+1
        segmen[segmen == 0 ] = count_label
        
        return count_label
    
    def extract_region (self,segmen,image):
        list_region = []
        list_neighbor_pair = []
        
        for region in measure.regionprops(segmen):
            img_segmen = color.gray2rgb(region.image) * image[
                    region.bbox[0]:region.bbox[2],
                    region.bbox[1]:region.bbox[3]]
            
            self.hierarchy.append(region.label)
            r = Region(region.label,region.bbox,img_segmen)
            
            list_neighbor_pair = list_neighbor_pair + self.get_neighbor(segmen,r)
            
            list_region.append(r)
        
        return np.array(list_region),np.array(list_neighbor_pair)
        
    def get_neighbor(self,segmen,region):
        label_region = region.label
        bbox_region = region.bbox
        neighbor_list = []
        
        min_y, min_x, max_y, max_x =  bbox_region
        
        
        if bbox_region[0] > 1:
            min_y = min_y - 2
        
        if bbox_region[1] > 1:
            min_x = min_x - 2
        
        if bbox_region[2] < segmen.shape[0]-1:
            max_y = max_y + 2
        
        if bbox_region[3] < segmen.shape[1]-1:
            max_x = max_x + 2
            
        #print(min_y, min_x, max_y, max_x)
        region = segmen[min_y:max_y,min_x:max_x]
        
        neighbor = np.copy(region)
        neighbor[neighbor != label_region] = 0
        
        #neighbor = dilation(neighbor, square(3))
        neighbor = find_boundaries(neighbor, connectivity=1, mode='thick', background=0).astype(int)
    
        neighbor = np.where(neighbor == 1)
        neighbor = np.unique(region[neighbor])
        
        
        for j in neighbor:
            if j != label_region:
                neighbor_list.append([label_region,j])
        
        return neighbor_list

    def mergebbox(self,bboxRegion1,bboxRegion2):
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
    
    def similarity_neigh_pair(self,list_neighbor_pair,list_region,segmen ):
        list_similarity = []
        for i in range(len(list_neighbor_pair)):
            ri = list_region[ list_neighbor_pair[i][0] - 1 ] 
            rj = list_region[ list_neighbor_pair[i][1] - 1 ] 
            
            list_similarity.append( similarity_neighbor( ri,rj ,segmen ) )
        return np.array(list_similarity)
    
    def merge_region(self,image):
        
        
        segmen = self.segmen

        list_similarity = self.list_similarity
        list_neighbor_pair = self.list_neighbor_pair
        list_region = self.list_region 
        max_label = self.max_label
        
        temp_hierarchy = []
        hierarchy = self.hierarchy
        
        
        while (len(list_similarity) > 0) :
            
            print(hierarchy)
            max_index = np.argmax(list_similarity)
            
            
            ri,rj = list_neighbor_pair[max_index][0],list_neighbor_pair[max_index][1]
            
            neighbor_index = np.concatenate((
                    np.where(list_neighbor_pair[:,0] == ri),
                    np.where(list_neighbor_pair[:,0] == rj)),axis = 1 )
            
            
            list_neighbor_pair = np.delete(list_neighbor_pair,neighbor_index[0],axis = 0)
            
            if( ri in hierarchy):
                hierarchy.remove(ri)
            
            if( rj in hierarchy):
                hierarchy.remove(rj)
            
            #print(type(self.hierarchy))
            
            
            list_similarity = np.delete(list_similarity,neighbor_index[0],axis = 0)
            
            other_neigh_index = np.concatenate((
                    np.where(list_neighbor_pair[:,1] == ri),
                    np.where(list_neighbor_pair[:,1] == rj)),axis = 1 )
            
            other_neigh = list_neighbor_pair[other_neigh_index[0]]
            
            
            list_neighbor_pair = np.delete(list_neighbor_pair,other_neigh_index[0],axis = 0)
            list_similarity = np.delete(list_similarity,other_neigh_index[0],axis = 0) 
            
            max_label = max_label + 1 
    
            rt_label = max_label
            temp_hierarchy.append(rt_label)
            
            if(len(hierarchy) < 1 ):
                hierarchy = temp_hierarchy
                temp_hierarchy = []
                
                self.img_hierarchy.append(segmen)
            
            
            rt_bbox =  self.mergebbox(list_region[ri-1].bbox,list_region[rj-1].bbox)
            #segmen2 = np.copy(segmen)
            
            
            
            segmen[segmen == ri ] = rt_label
            segmen[segmen == rj ] = rt_label
            
            rt_image = np.copy(segmen[rt_bbox[0]:rt_bbox[2],rt_bbox[1]:rt_bbox[3]])
            rt_image[rt_image != rt_label ] = 0
            rt_image[rt_image == rt_label ] = 1
            
            
            
            rt_image = color.gray2rgb(rt_image) * image[rt_bbox[0]:rt_bbox[2],rt_bbox[1]:rt_bbox[3]] 
            
            r = Region(rt_label,rt_bbox,rt_image)
            
            list_region = np.append(list_region,r)
            
            rt_neighbor = self.get_neighbor(segmen,r)
            
            if(len(rt_neighbor) > 0):
            
                other_neigh = np.unique(other_neigh[:,0])
                new_other_neigh = np.zeros((other_neigh.shape[0],2), dtype=int)
                new_other_neigh[:,0] = other_neigh 
                new_other_neigh[:,1] = rt_label  
                
                #print(max_label," : ", rt_neighbor,"\n",new_other_neigh)
                
                new_neighbor = np.concatenate((rt_neighbor,new_other_neigh), axis = 0)
                
                
                new_sim = []
                
                for i in range(len(new_neighbor)):
                    
                    ri = list_region[ new_neighbor[i][0] - 1 ] 
                    rj = list_region[ new_neighbor[i][1] - 1 ] 
                    
                    new_sim.append( similarity_neighbor( ri,rj ,segmen ) )
                
            
                list_neighbor_pair = np.append(list_neighbor_pair,new_neighbor,axis = 0)
                list_similarity = np.append(list_similarity,new_sim) 
            
            print(list_neighbor_pair)
            print("")


    