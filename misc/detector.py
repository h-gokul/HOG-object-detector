
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm 
import os

import args as params
from utils import load_dataset, get_hog_descriptors, get_class_labels, rescale, ImageData
from sliding_window import *
import pickle
from preprocess import readlabel



def topkboxes(detections, k=5, c_thresh = 0.7): 
    if k==1: return detections[np.argmax(detections[:, 1])]
    
    k_idxs = np.argsort(detections[:, 1])[-k:]
    topk = detections[k_idxs] # find best k
    bestk = np.where(topk[:, 1] > c_thresh) # choose only predictions >0.7
    if len(bestk)==0: return topk[np.argmax(topk[:, 1])] # if low confidence, return k
    return topk[bestk]

def mse_error(gt, preds):
    return np.sqrt(np.sum((gt-preds)**2))

def detector(img, model):
    detections = []
    multi_scales = [ 1.0, 1.25, 1.5] # [ 0.75, 1.0, 1.25, 1.5]

    ## for a range of different image scales in an image pyramid
    for current_scale in multi_scales:
        print(f"scale: {current_scale}")
        resized_img = rescale(img.copy(), current_scale)

        window_size = params.DATA_WINDOW_SIZE
        step = math.floor(resized_img.shape[0] / 64)
        if not (step > 0): continue    
        
        # Across each scan window
        window_slider = tqdm(sliding_window(resized_img, window_size, step_size=step))
        for (x, y, window) in window_slider: 
                    
            # for each window region get the HOG feature point descriptors
            img_data = ImageData(window)
            img_data.compute_hog_descriptor() # note this image is 2.56x zoomed
            
            # generate and classify each window by constructing a HOG
            # descriptor and passing it through the SVM classifier
            if img_data.hog_descriptor is not None:
                # print("detecting with SVM ...")
                testdata = np.float32([img_data.hog_descriptor[:,0]])
                class_probability = model.predict_proba(testdata)
                pred_class = np.argmax(class_probability)
                
                # if we get a detection, then record it
                if pred_class == params.DATA_CLASS_NAMES["phone"]:
                    # store rect as (x1, y1) (x2,y2) pair
                    rect = np.float32([x, y, x + window_size[0], y + window_size[1]])
                    
                    rect /= current_scale
                    detections.append((rect, class_probability[0][pred_class]))
    return np.array(detections)



if __name__ == "__main__":

    # load SVM from file
    try:
        model = pickle.load(open(params.HOG_SVM_PATH, 'rb'))
    except:
        print("[ERROR] Missing files - SVM!");
        print("[ERROR] Have you performed trained SVM to produce these files ?");
        
    lines = readlabel(params.label_path)[: 20]

    N = len(lines)
    correct = 0
    for i, line in enumerate(lines):

        name, x, y =  line.rsplit(' ')
        print(f"############# Image {name} #############")
        x, y = float(x), float(y) 
        img = cv2.imread(f'{params.data_path}/{name}')
        h,w = img.shape[:2]
        cx, cy = int(x*w), int(y*h)

        # detect all boxes
        detections = detector(img, model)

        # take top k boxes 
        topk_detections = topkboxes(detections)

        # compute centroid of k boxes
        box = np.mean(topk_detections[:,0], axis=0) # regular mean of k boxes
        pred_cx, pred_cy = np.mean((box[0], box[2])), np.mean((box[1], box[3])) # center

        gt = np.array([cx/w, cy/h])
        preds = np.array([pred_cx/w, pred_cy/h])

        if mse_error(gt,  preds) < 0.05 :
            correct+=1

    print(f"Accuracy = {correct/N},  {correct}/{N}")