# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:20:58 2021

@author: Somanshu
"""

import cv2
from torch.utils.data import Dataset
import json
import os

class Pedestrian_Dataset(Dataset):
    
    def __init__(self,root,test_json):
        
        f = open(test_json)
        test_files = json.load(f)
        self.image_names = test_files['images']
        self.labels = test_files['annotations']
        self.root = root
        
    def __getitem__(self,index):    
        
        image = cv2.imread(os.path.join(self.root,self.image_names[index]['file_name']))
        img_id = self.image_names[index]['id']
        return image,img_id
    
    def __len__(self):
        return len(self.image_names)