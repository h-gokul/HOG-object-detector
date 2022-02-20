import time

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


def detector2(model, img):
    start = time.time()
    detections = []
    multi_scales = [ 1.0, 1.25] #, 1.5 [ 0.75, 1.0, 1.25, 1.5]
    win_samples = {'feature' : [], 'xy' : [], 'scale' : [] }
    ## for a range of different image scales in an image pyramid
    for current_scale in multi_scales:
        print(f"scale: {current_scale}")
        
        resized_img = rescale(img.copy(), current_scale)

        window_size = params.DATA_WINDOW_SIZE
        step = math.floor(resized_img.shape[0] / 64)
    
        # if not (step > 0): continue [UNCOMMENT]    

        # Across each scan window
        window_slider = tqdm(enumerate(sliding_window(resized_img, window_size, step_size=step)))
        for i, (x, y, window) in window_slider:         
            # for each window region get the HOG feature point descriptors
            img_data = ImageData(window)
            img_data.compute_hog_descriptor() # note this image is 2.56x zoomed

            # classify each window's HOG descriptor by passing it through the SVM classifier
            if img_data.hog_descriptor is not None:
                descriptor = np.float32([img_data.hog_descriptor[:,0]])
                win_samples['feature'].append(descriptor.squeeze())
                win_samples['xy'].append((x, y))
                win_samples['scale'].append(current_scale)
        logger.write(f"{i} windows at scale: {current_scale} \n")
    print(f"total number of boxes: {len(np.array(win_samples['scale']))}")

    logger.write(f"total number of boxes: {len(np.array(win_samples['scale']))} \n")
    scales = np.array(win_samples['scale'])
    testdata = np.array(win_samples['feature'])

    print("Predicting With SVM"); logger.write("Predicting With SVM \n")
    class_probabilities = model.predict_proba(testdata)
    pred_classes = np.argmax(class_probabilities, axis=1)
    idxs = np.where(pred_classes == params.DATA_CLASS_NAMES["phone"])[0]
    confidences = class_probabilities[idxs][:, params.DATA_CLASS_NAMES["phone"]]
    locations = np.array(win_samples['xy'])[idxs]
    scales = np.array(win_samples['scale'])[idxs][:, np.newaxis]
    xmin = locations[:,0][:, np.newaxis]
    ymin = locations[:,1][:, np.newaxis]
    xmax = xmin + window_size[0] 
    ymax = ymin + window_size[1]

    rectangles = np.hstack((xmin/scales,ymin/scales, xmax/scales, ymax/scales)) 
    detections = np.hstack([rectangles, confidences[:, np.newaxis]])
    time_elapsed = time.time()-start
    print(f" Total time for detection: {time_elapsed} \n")
    logger.write(f" Total time for detection: {time_elapsed} \n")
    return detections

def topkboxes(detections, k=5, c_thresh = 0.7): 
    if k==1: return detections[np.argmax(detections[:, 4])]
    k_idxs = np.argsort(detections[:, 4])[-k:]
    topk = detections[k_idxs] # find best k
    bestk = np.where(topk[:, 4] > c_thresh) # choose only predictions >0.7
    if len(bestk)==0: return topk[np.argmax(topk[:, 4])] # if low confidence, return k
    return topk[bestk]

def mse_error(gt, preds):
    return np.sqrt(np.sum((gt-preds)**2))

logger = open("detection_results.txt", "a") # global declaration

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
    for i, line in tqdm(enumerate(lines)):
        name, x, y =  line.rsplit(' ')
        print(f"############# Image {name} #############")
        logger.write(f"############# Image {name} #############\n")
        x, y = float(x), float(y) 
        img = cv2.imread(f'{params.data_path}/{name}')
        h,w = img.shape[:2]
        cx, cy = int(x*w), int(y*h)
        
        detections = detector2(model, img)
        topk_detections = topkboxes(detections)
        boxes = topk_detections[:, :4]
        box = np.mean(boxes, axis=0) # regular mean of k boxes
        pred_cx, pred_cy = np.mean((box[0], box[2])), np.mean((box[1], box[3])) # center

        gt = np.array([cx/w, cy/h])
        preds = np.array([pred_cx/w, pred_cy/h])

        if mse_error(gt,  preds) < 0.05 :
            print("Correct prediction")
            logger.write(f"{i} Correct prediction \n\n")
            correct+=1
        else:
            print("Wrong prediction")
            logger.write(f"{i} Wrong prediction \n\n")

    print(f"Accuracy = {correct/N},  {correct}/{N}")
    logger.write(f"Accuracy = {correct/N},  {correct}/{N}")
    logger.close()