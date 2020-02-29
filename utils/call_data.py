import cv2
import numpy as np
import glob

def call_data_path(dir_path):
    return glob.glob(dir_path + '*.*')

def img_size_check(img_array):
    height_resize = 0
    width_resize = 0
    
    temp = list(img_array)
    
    for i in range(len(temp)):
        height_resize = temp[i].shape[0] % 4
        width_resize = temp[i].shape[1] % 4
        
        if height_resize != 0 or width_resize != 0:
            width = temp[i].shape[1] + (4 - width_resize)
            height = temp[i].shape[0] + (4 - height_resize)
            
            temp[i] = cv2.resize(temp[i], dsize=(width, height))
    
    img_array = np.array(temp)
    
    return img_array

def read_batch(batch_idx, batch_size, img_list, label_list, random_indexes):
    trn_img = []
    trn_label = []
    
    batch_cnt = batch_idx * batch_size
    
    for random_idx in random_indexes[batch_cnt : (batch_cnt + batch_size)]:
        trn_img.append(cv2.imread(img_list[random_idx], cv2.IMREAD_COLOR))
        trn_label.append(cv2.imread(label_list[random_idx], cv2.IMREAD_GRAYSCALE))
        
    trn_img = np.array(trn_img)
    trn_label = np.array(trn_label)
    
    trn_img = img_size_check(trn_img)
    trn_label = img_size_check(trn_label)
    
    return trn_img, trn_label