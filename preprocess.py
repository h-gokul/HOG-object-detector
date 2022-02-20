import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 
import os
import args as params

def foldercheck(Savepath):
    if(not (os.path.isdir(Savepath))):
        print(Savepath, "  was not present, creating the folder...")
        os.makedirs(Savepath)

def window_limit(cx, cy, x_win, y_win, h, w):    
    x_min,y_min = int(max(0, cx - x_win//2)) , int(max(0, cy - y_win//2))
    x_max, y_max = int(min(cx + x_win//2, w)), int(min(cy + y_win//2, h))
    return x_min,y_min, x_max, y_max

def readlabel(label_path):
    with open(label_path) as f:
        lines = f.read().splitlines()    
    f.close()    
    return lines

def extract_samples(data_path, x_win=50, y_win = 50):
    """
    Extracts positive (phone) and negative (background) patches
    """
    foldercheck(f"{data_path}/test_pos")
    foldercheck(f"{data_path}/test_neg")
    foldercheck(f"{data_path}/train_pos")
    foldercheck(f"{data_path}/train_neg")
    
    for mode in ['train', 'test']:
        if mode == 'train': 
            lines = readlabel(params.label_path)[20:]
        else:
            lines = readlabel(params.label_path)[:20]

        for i, line in tqdm(enumerate(lines)):
            name, x, y =  line.rsplit(' ')
            x, y = float(x), float(y) 
            image = cv2.imread(f'{data_path}/{name}')
            h,w = image.shape[:2]
            cx, cy = int(x*w), int(y*h)
            # extract the positive sample's patch
            x_min,y_min, x_max, y_max = window_limit(cx, cy , x_win, y_win, h, w)
            pos_patch =  cv2.resize(image[y_min: y_max, x_min: x_max], (x_win, y_win), interpolation = cv2.INTER_AREA)
            
            # choose a value for negative sample without inlcuding positve patch areas
            range_x, range_y = np.arange(w), np.arange(h) 
            bg_xrange = np.hstack((range_x[x_win:x_min], range_x[x_max:-x_win]))
            bg_yrange = np.hstack((range_y[y_win:y_min], range_y[y_max:-y_win]))
            bg_x = np.random.choice(bg_xrange,1 )[0]
            bg_y = np.random.choice(bg_yrange,1 )[0]

            # extract the patch    
            bx_min,by_min, bx_max, by_max = window_limit(bg_x, bg_y , x_win, y_win, h, w)
            neg_patch =  cv2.resize(image[by_min: by_max ,bx_min: bx_max], (x_win, y_win), interpolation = cv2.INTER_AREA)

            cv2.imwrite( f"{data_path}/{mode}_pos/{name}", pos_patch)
            cv2.imwrite( f"{data_path}/{mode}_neg/{name}", neg_patch)
    
