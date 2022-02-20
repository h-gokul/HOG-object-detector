# HOG Object Detection:
This repository contains an Object detector based on Histogram of Oriented Gradients Descriptor 

This object detector slides a window of fixed size= (50,50) across the image and classifies whether the image is the object exists or not
This exhaustive sliding window search is repeated for 3 different scales [1.0, 1.25, 1.5] in a image pyramid.
Classification is done using Support vector machines (SVM)


## Training:
We first extract the positive and negative image patch samples from the dataset. The function `extract_samples` does this.
Next we train the SVM with the given input samples.
Run the below command to train the model. 
`python3 train.py`

## Testing:
Run the below command to test the model that we trained. 
`python3 train.py`

## Detection:
To detect the center of the mobile phone:
- we perform an exhaustive sliding window search on a image pyramid and collect all the windows
- extract HOG descriptors from each window and classify whether it is an object or not.
- retrieve the windows predicted to contain the object along with confidence scores.
- Sort the bounding-box based on confidence scores, and select the top k boxes. Also, ensure the confidence >70% (k=5 here)
- The predicted centroid is  the centroid the top k boxes.


### Future Improvements: 
[TODO]: Tune window size. Right now it is a rigid square.
[TODO]: Data Augmentation for training SVM
[TODO]: Negative Hard mining to training SVM
[TODO]: Latent SVM approach, as approached in Deformable Part Models



