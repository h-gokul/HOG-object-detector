import cv2
#################################################################################
# settings for datasets in general
data_path = "./find_phone"
label_path = data_path + '/labels.txt'

process = True

train_paths = [data_path+'/train_neg', data_path+'/train_pos']
test_paths = [data_path+'/test_neg', data_path+'/test_pos']

x_win, y_win = 50, 50
DATA_WINDOW_SIZE = [x_win, y_win]
sampling_sizes = [0 , 0]
sample_from_centre = [True, True]
DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 0
DATA_CLASS_NAMES = { "other": 0, "phone": 1}

DATA_WINDOW_SIZE = [50, 50]

HOGscale_factor = 2.56 # size to scale the image so that HOG descriptor works

HOG_SVM_PATH = "./pretrained/svm_hog.sav"

