# Train an SVM using HOG features
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 
import os
import args as params
from utils import load_dataset, get_hog_descriptors, get_class_labels, print_duration
import pickle
from sklearn import svm

def SVMtrainer(train_samples, train_labels):

    print("Training SVM...")
    start_time = cv2.getTickCount()

    # define SVM
    model = svm.SVC(probability=True) # rbf kernel svm
    # fit the model
    model.fit(train_samples, train_labels)

    # save the tained SVM to file so that we can load it again for testing / detection
    pickle.dump(model, open(params.HOG_SVM_PATH, 'wb'))

    ############################################################################
    # measure performance of the SVM trained on the bag of visual word features
    preds = model.predict(train_samples) 
    # probabilities = model.predict_proba(train_samples)

    # perform prediction over the set of examples we trained over
    error = (np.absolute(train_labels - preds).sum()) / float(preds.shape[0])
    
    # we are succesful if our prediction > than random
    # e.g. for 2 class labels this would be 1/2 = 0.5 (i.e. 50%)

    if error < (1.0 / len(params.DATA_CLASS_NAMES)):
        print("Trained SVM done \n Training set error: {}% ".format(round(error * 100,2)))
        print("-- Model Accuracy {}% !".format(round((1.0 - error) * 100,2)))
    else:
        print("Failed to train SVM. {}% error".format(round(error * 100,2)))

    print_duration(start_time)
    

if __name__ == '__main__':
    # process the data if not split to postiive and negative patches
    if params.process :
        from preprocess import extract_samples
        extract_samples(params.data_path, x_win=params.x_win, y_win = params.y_win)
    
    # load training dataset
    train_dataset= load_dataset(params)

    # Compute Descriptors
    for data in train_dataset:
        data.compute_hog_descriptor()

    # Load the descriptors and class labels
    train_samples = get_hog_descriptors(train_dataset).squeeze()
    train_labels = get_class_labels(train_dataset)

    # SVM training..
    SVMtrainer(train_samples, train_labels)

    