# -*- coding: utf-8 -*-
"""
Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import os
import cv2
import torch
import torchvision
import argparse
import json
import numpy as np
from imutils.object_detection import non_max_suppression
from torch.utils.data import DataLoader
from dataset_class import Pedestrian_Dataset
from utils import get_results_in_json

def get_output(model_out,image_id,show,image,show_path):
    
    img_id = image_id.item()
    labels = model_out[0]['labels']
    bboxs = model_out[0]['boxes']
    scores = model_out[0]['scores']
    person_indices = (labels==1).nonzero(as_tuple=True)
    person_indices = person_indices[0]
    labels= labels.index_select(0,person_indices)
    scores= scores.index_select(0,person_indices)
    bboxs= bboxs.index_select(0,person_indices)
    
    labels = labels.detach().numpy()
    scores = scores.detach().numpy()
    bboxs = bboxs.detach().numpy()
    
    boxes_nms = non_max_suppression(bboxs, probs=scores, overlapThresh=0.9)
    
    scores_nms = None
    First = True
    for box in boxes_nms:
        ind = np.where(np.all(bboxs.astype(int)==box,axis=1))[0]
        if First:
            scores_nms=scores[ind]
            First = False
        else: 
            scores_nms = np.vstack((scores_nms,scores[ind]))
      
    bboxs = boxes_nms
    bboxs = np.array(bboxs)
    if len(bboxs)!=0:
        bboxs[:,2] = bboxs[:,2]-bboxs[:,0]
        bboxs[:,3] = bboxs[:,3]-bboxs[:,1]
    scores= np.array(scores_nms)
    bboxs = bboxs.astype(float)

    cond = np.where(scores<0.7)[0]
        #print(scores)
    scores = np.delete(scores,cond,axis=0)
    bboxs = np.delete(bboxs,cond,axis=0)
    

    if show:
        for (x,y,w,h) in bboxs:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite(show_path+"/"+str(img_id)+".png",image)
            
        
    res = {
            "image_id": img_id,
            "bbox": bboxs,
            "scores": scores
            }
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='Getting the test data')
    parser.add_argument('-r', '--root', type=str, default='pedestrian_detection', required=True, help="Path to the dataset root directory")
    parser.add_argument('-t', '--test', type=str, default='test', required=True, help="Path to the test json")
    parser.add_argument('-o', '--out', type=str, default='out', required=True, help="Path to the output json")
    parser.add_argument('-m', '--model', type=str, default='model', required=True, help="Path to pretrained Faster RCNN weights file")
    args = parser.parse_args()
    return args

def inference(model,TestLoader,device,verbose,show,show_path):
    
    results = []
    model.eval()
    with torch.no_grad():  
        
        #id_list = []
        for index,(x,img_id) in enumerate(TestLoader):
            #print(var)
            x.to(device)
            image_show = x
            
            image_show = torch.reshape(image_show,(image_show.shape[1],image_show.shape[2],image_show.shape[3]))
            image_show = image_show.detach().numpy()
            
            num_images = x.shape[0]
            X = []
            
            for i in range(num_images):
                image = (x[i]/255)
                image = torch.swapaxes(image,0,2)
                image = torch.swapaxes(image,1,2)
                X.append(image)
            
            #X.to(device)
            output = model(X)
            
            if show:
                if not os.path.isdir(show_path):
                    os.mkdir(show_path)
            
            output=get_output(output,img_id,show,image_show,show_path)
            if verbose:
                print("Image "+ str(index) + " has been processed by FRCNN pipeline")
            results.append(output)
            
    return results


def pedestrian_detection(root,test_json,output_path,weight_path,show,show_path,verbose):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,progress=True,pretrained_backbone=True)
    model.to(device)
    
    
    TestSet = Pedestrian_Dataset(root, test_json)
    TestLoader = DataLoader(TestSet, batch_size=1, shuffle=True, num_workers=2)
        
    results = inference(model,TestLoader,device,verbose,show,show_path)
    result_json = get_results_in_json(results)
    
    
    with open(output_path, 'w') as fout:
        json.dump(result_json, fout)
    
    #TODO: GPU testing
    
    return model


if __name__ == "__main__":  
    
    args = parse_args()   
    
    root = args.root
    test_json = args.test
    output_path = args.out
    weight_path = args.model
    verbose = False
    show = False
    show_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_FRCNN/predicted_masks"
    model=pedestrian_detection(root, test_json, output_path, weight_path, show, show_path,verbose)
    
"""
root = "./pedestrian_detection"
output_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_FRCNN/output.json"
test_json = "./pedestrian_detection/PennFudanPed_full.json"
show= True
show_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_FRCNN/predicted_masks"
out_path = "./pedestrian_detection/PennFudanPed/Output_Pretrained_FRCNN/wts.pt"
"""