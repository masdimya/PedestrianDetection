# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:10:38 2019

@author: user
"""
import os
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries
from skimage import measure, color
from skimage.segmentation import  felzenszwalb, mark_boundaries


from similarity import similarity_neighbor



class Region():
    def __init__(self,label,bbox,segmen_area,mask):
        self.label=label
        self.bbox=bbox
        self.image = segmen_area
        self.mask = mask
        self.neighbor = []
        
        
class selective_search():
    def __init__(self,image):
        
        self.image = image
        '''
            felzenszwalb superpixel segmentation 
        '''
        self.segmen = self.segmentation(image)
        
        
        '''
            slic superpixel segmentation 
        '''
        
        self.max_label = self.fix_label(self.segmen)
        
        
        
        self.sizeImage = image[:,:,0].shape
        
        self.matric_pair = np.zeros((self.max_label,self.max_label))
        self.neighbor_matric = np.zeros((self.max_label,self.max_label), dtype = bool)
        
        self.img_hierarchy = [np.copy(self.segmen)]
        
        self.list_region = []
        self.list_neighbor_pair = []
        self.boundingbox = []
        #self.image_crop = []
        
        
        
        
        self.extract_region (self.segmen,image)
        
        
        self.similarity_neigh_pair(self.list_neighbor_pair, 
                                   self.list_region,
                                   self.segmen)
        
        self.merge_region(image)
        
        self.more_region(image,self.list_region)
        
        self.boundingbox = np.unique(self.boundingbox, axis = 0)
        self.boundingbox = self.del_large_bbox(self.boundingbox)
        
#        self.save_boundaries(image)
        
#        self.save_label2rgb(image)
        self.save_crop(image)
    
    def segmentation(self,image):
        
        kondisi = True
        min_size = 600
        while kondisi :
            seg = felzenszwalb(image, scale = 1, sigma = 0.8, min_size = min_size  )
            
            if seg.max() > 100 and seg.max() < 120 :
                kondisi = False
            else :
                min_size += 200
        return seg
                
    
    def fix_label (self,segmen) :
        count_label = segmen.max()
        
        count_label = count_label+1
        segmen[segmen == 0 ] = count_label
        
        return count_label
    
    def extract_region (self,segmen,image):
        
        print("ekstrak region .... ")
        list_region = []
        list_neighbor_pair = []
        list_bounding = []
        list_image = []
        
        for region in measure.regionprops(segmen):
            img_segmen = color.gray2rgb(region.image) * image[
                    region.bbox[0]:region.bbox[2],
                    region.bbox[1]:region.bbox[3]]
            
            list_bounding.append(list(region.bbox))
            #list_image.append(self.crop_image(image,region.bbox))
            r = Region(region.label,region.bbox,img_segmen,region.image)
            
            list_neighbor_pair = list_neighbor_pair + self.get_neighbor(segmen,r)
            
            list_region.append(r)
        
        print(len(list_region))
        print("ekstrak region selesai.... \n")
        
        self.list_region = np.array(list_region)
        self.list_neighbor_pair = np.array(list_neighbor_pair)
        self.boundingbox = list_bounding
#        self.image_crop = list_image
        
        
        
    def get_neighbor(self,segmen,reg_):
        label_region = reg_.label
        bbox_region = reg_.bbox
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
                reg_.neighbor.append(j)
                neighbor_list.append([label_region,j])
        
        return neighbor_list
    
    def fast_get_neighbor(self,rt_label,i,j):
        neighbor = self.neighbor_matric[:,i]+self.neighbor_matric[:,j]
        
        indices = np.where(neighbor == True)
        
        pair = np.zeros((len(indices[0]),2),dtype=int)
        
        pair[:,0] = rt_label
        pair[:,1] = indices[0]+1
        
        
        
        
        return pair

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
        
        return [BBox_Ymin,BBox_Xmin,BBox_Ymax,BBox_Xmax]
    
    def similarity_neigh_pair(self,list_neighbor_pair,list_region,segmen ):
        for i in range(len(list_neighbor_pair)):
            ri_index = list_neighbor_pair[i][0] - 1
            rj_index = list_neighbor_pair[i][1] - 1
            
            ri = list_region[ ri_index ] 
            rj = list_region[ rj_index ] 
            
            #print('sim antara ri {} dan rj {} : {}'.format(ri.label,rj.label,similarity_neighbor( ri,rj ,segmen )))
            self.matric_pair[ri_index ,rj_index] = similarity_neighbor( ri,rj ,segmen )
            self.neighbor_matric[ri_index ,rj_index] = True
        
    def crop_image(self,image,bbox):
        min_r, min_c, max_r, max_c = bbox 
        
        wrap = cv2.resize(image[min_r:max_r, min_c:max_c], dsize = (64,128), interpolation = cv2.INTER_CUBIC)
        
        return wrap
    def merge_region(self,image):
        
        
        segmen = self.segmen


        max_label = self.max_label
        

        
        count = 0
        
        max_sim = 1
        
        while (max_sim > 0) :
            

            
            print("iter : ",count," Mencari max value sim ")
            

            ri_index,rj_index = np.unravel_index(self.matric_pair.argmax(), 
                                                 self.matric_pair.shape)
            
            ri,rj = ri_index+1,rj_index+1
            
#            print(ri,rj)

            max_sim = self.matric_pair[ri_index,rj_index]

            
            print("iter : ",count," delete region pair ")
            
            
            self.matric_pair[ri_index,:] = 0
            self.matric_pair[rj_index,:] = 0
            
            self.neighbor_matric[ri_index,:] = False
            self.neighbor_matric[rj_index,:] = False
            
            
            
            print("iter : ",count," delete semilarity region pair ")
            
            self.matric_pair[:,ri_index] = 0
            self.matric_pair[:,rj_index] = 0
            
            

            
            print("iter : ",count," delete  region pair milik tetangga")

            max_label = max_label + 1 
    
            rt_label = max_label
            
            
            print("iter : ",count," membuat rt ")
            
            rt_bbox =  self.mergebbox(self.list_region[ri-1].bbox,self.list_region[rj-1].bbox)
            
            
            
            segmen[segmen == ri ] = rt_label
            segmen[segmen == rj ] = rt_label
            
            #percent = (self.max_label*10)//100
            #if(percent % count == 0):
            self.img_hierarchy.append(np.copy(segmen))
            
            rt_image = np.copy(segmen[rt_bbox[0]:rt_bbox[2],rt_bbox[1]:rt_bbox[3]])
            rt_image[rt_image != rt_label ] = 0
            rt_image[rt_image == rt_label ] = 1
            
            rt_mask = np.copy(rt_image)
            
            
            rt_image = color.gray2rgb(rt_image) * image[rt_bbox[0]:rt_bbox[2],rt_bbox[1]:rt_bbox[3]] 
            
            
            r = Region(rt_label,rt_bbox,rt_image,rt_mask)
            
            
            
            matric_x = self.matric_pair.shape[1]
            self.matric_pair = np.append(self.matric_pair,
                                         np.zeros((1,matric_x)) , axis = 0)
            
            self.neighbor_matric = np.append(self.neighbor_matric,
                                         np.zeros((1,matric_x) ,dtype = bool) , axis = 0)
            
            
            
            
            matric_y = self.matric_pair.shape[0]
            self.matric_pair = np.append(self.matric_pair,
                                         np.zeros((matric_y,1)) , axis = 1)
            self.neighbor_matric = np.append(self.neighbor_matric,
                                         np.zeros((matric_y,1) ,dtype = bool) , axis = 1)
            
#            print("iter : ",count," mencari tetangga rt ")
            
         
#            rt_neighbor = self.get_neighbor(segmen,r)
            fast_rt_neighbor = self.fast_get_neighbor(rt_label,ri_index,rj_index)
            
            
            
#            rt_neighbor = self.fast_get_neighbor(self.list_region[ri_index],self.list_region[rj_index],rt_label)
            
           
            
            if(len(fast_rt_neighbor) > 0):
                
#                print("iter : ",count," menambahkan tetangga rt  ")
                
#                r.neighbor = rt_neighbor[:,1].tolist()
                
                
                self.list_region = np.append(self.list_region,r)
                self.boundingbox.append(rt_bbox)
#                self.image_crop.append(self.crop_image(image,r.bbox))
                
                
                for i in range(len(fast_rt_neighbor)):
                    ri_index = fast_rt_neighbor[i][0] - 1
                    rj_index = fast_rt_neighbor[i][1] - 1
                    
                    ri = self.list_region[ ri_index ] 
                    rj = self.list_region[ rj_index ] 
                    
                    self.matric_pair[ri_index,rj_index] =  similarity_neighbor( ri,rj ,segmen ) 
                    
                    self.neighbor_matric[ri_index,rj_index] =  True
                    self.neighbor_matric[rj_index,ri_index] =  True
                
                
            
                print("iter : ",count," finish rt ")
            else :
                max_sim = 0
            
            
            
            
            count = count + 1
    
    def more_region(self,image,list_region):
        
        
        
        sizeimg = self.sizeImage[0] * self.sizeImage[1]
        
        
        
        for region in list_region:
            regionsize = (np.sum(region.mask.astype(int)) / sizeimg ) *  100
            
            if regionsize > 40 and regionsize < 90:
                
                
                matriks_one = np.full(self.sizeImage,1)
                matriks_seg = np.zeros(self.sizeImage,dtype = int)
                
                matriks_seg[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = region.mask
                new_seg = matriks_one - matriks_seg 
                
                label = measure.label(new_seg)
                
                for new_region in measure.regionprops(label):
#                    self.image_crop.append(self.crop_image(image,new_region.bbox))
                    self.boundingbox.append(new_region.bbox)
                
    def del_large_bbox(self,bbox):
        sizeimg = self.sizeImage[0] * self.sizeImage[1]
        temp = []
        for i in bbox :
            area = (((i[2]-i[0]) * (i[3]-i[1]))/sizeimg) * 100
            if area < 60:
                temp.append(i)
        return temp
            
            
    def save_boundaries(self,image):
        currentDT = datetime.datetime.now()
        
        path  = currentDT.strftime("%Y-%m-%d_%H.%M.%S")
        os.mkdir(path)
         
        for i in range (len(self.img_hierarchy)):
            boundaries = mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),self.img_hierarchy[i])
            plt.imsave(path+"/segmen_{}.jpg".format(i),boundaries)
        
        os.mkdir(path+'_more')
    
    def save_label2rgb(self,image):
        currentDT = datetime.datetime.now()
        
        path  = currentDT.strftime("%Y-%m-%d_%H.%M.%S")
        os.mkdir(path)
         
        for i in range (len(self.img_hierarchy)):
            boundaries = color.label2rgb(self.img_hierarchy[i])
            plt.imsave(path+"/segmen_{}.jpg".format(i),boundaries)
        
        os.mkdir(path+'_more')
        
    
    def save_crop(self,image):
        currentDT = datetime.datetime.now()
        
        path  = currentDT.strftime("%Y-%m-%d_%H.%M.%S")
        os.mkdir(path+'_crop')
         
        for i in range (len(self.boundingbox)):
            crop = self.crop_image(image,self.boundingbox[i])
            rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            plt.imsave(path+"_crop/segmen{}.jpg".format(i),rgb_img)
    
    def image_crop(self):
        list_img = list()
        for i in range (len(self.boundingbox)):
            crop = self.crop_image(self.image,self.boundingbox[i])
            rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            list_img.append(rgb_img)
        return list_img

        
    