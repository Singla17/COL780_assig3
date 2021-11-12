# -*- coding: utf-8 -*-
"""
Somanshu 2018EE10314
Lakshya  2018EE10222
"""

def IOU(bbox_gt,bbox_pred):
    """
    Inputs: bbox_gt- Is the bounding box which represents the ground truth
            bbox_pred- Is the predicted bounding box
            
    Outputs: score- Is the IOU score for the given set of inputs
    
    Works only when edges of the two boxes are parellel
    """

    if bbox_gt[0]<bbox_pred[0] and bbox_gt[2]<bbox_pred[0]:
        return 0.0
    elif bbox_pred[0]<bbox_gt[0] and bbox_pred[2]<bbox_gt[0]:
        return 0.0
    elif bbox_gt[1]<bbox_pred[1] and bbox_gt[3]<bbox_pred[1]:
        return 0.0
    elif bbox_pred[1]<bbox_gt[1] and bbox_pred[3]<bbox_gt[1]:
        return 0.0
    
    hori_dist = []
    verti_dist = []
    hori_dist.append(bbox_gt[0])
    hori_dist.append(bbox_gt[2])
    hori_dist.append(bbox_pred[0])
    hori_dist.append(bbox_pred[2])
    verti_dist.append(bbox_gt[1])
    verti_dist.append(bbox_gt[3])
    verti_dist.append(bbox_pred[1])
    verti_dist.append(bbox_pred[3])
    
    hori_dist.sort()
    verti_dist.sort()
    h_diff = hori_dist[2]-hori_dist[1]
    v_diff = verti_dist[2]-verti_dist[1]
    intersection = (h_diff+1)*(v_diff+1)
    
    area_1 = (bbox_gt[2]-bbox_gt[0]+1)*(bbox_gt[3]-bbox_gt[1]+1)
    area_2 = (bbox_pred[2]-bbox_pred[0]+1)*(bbox_pred[3]-bbox_pred[1]+1)
    
    iou = intersection / float(area_1+area_2-intersection)
    
    return iou

def get_results_in_json(results):
    
    result_json = []
    for result in results:
        
        image_id = result["image_id"]
        category_id = 1
        if result["scores"] is not None:
            for i in range(len(result["scores"])):
                score = result["scores"][i]
                if "boxes" in result:
                    bbox = result["boxes"][i]
                else :
                    bbox = result["bbox"][i]
                
                res = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox.tolist(),
                "score": score.item()
                }
                
                result_json.append(res)
            
    return result_json
            
        