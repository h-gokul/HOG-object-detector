
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 
import os
import args as params
from utils import load_dataset, get_hog_descriptors, get_class_labels
import pickle

if __name__ == "__main__":
    try:
        model = pickle.load(open(params.HOG_SVM_PATH, 'rb'))
    except:
        print("Missing files  SVM");
        print("-- have you performed training to produce this file ?")
        exit()

    # load testing data set in the same class order as training-  0: other: 1: phone

    test_dataset= load_dataset(params, mode='test')

    # Compute Descriptors
    for data in test_dataset:
        data.compute_hog_descriptor()

    # Load the descriptors and class labels
    test_samples = get_hog_descriptors(test_dataset).squeeze()
    test_labels = get_class_labels(test_dataset)

    print("Performing batch SVM classification over all data  ...")    
    preds = model.predict(test_samples) 
    # probabilities = model.predict_proba(train_samples)

    # compute and report the error over the whole set
    error = ((np.absolute(test_labels - preds).sum()) / float(preds.shape[0]))
    print(" Testing set error: {}% ".format(round(error * 100,2)))
    print("--  model accuracy {}% !".format(round((1.0 - error) * 100,2)))


