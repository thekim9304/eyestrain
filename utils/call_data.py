import glob
import cv2
import numpy as np

def call_data_path(directory_path):
    return glob.glob(directory_path + '/*.*')

def call_dataset(x_path, y_path):
    x_data = call_data_img(x_path)
    y_data = call_data_img(y_path, is_y=True)

    x_data = img_size_check2(x_data)
    y_data = img_size_check2(y_data)

    return x_data, y_data

def call_data_img(directory_path, is_y=False):
    img_ = []

    img_path = call_data_path(directory_path)

    if is_y == True:
        flag = cv2.IMREAD_GRAYSCALE
    else:
        flag = cv2.IMREAD_COLOR

    for i in img_path:
        img = cv2.imread(i, flag)
        
        if is_y == True:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] == 255:
                        img[i][j] = 1

        img_.append(img)

    img_ = np.array(img_)

    return img_

def img_size_check(img_array):
    height_resize = 0
    width_resize = 0
    
    print(img_array.shape)
    
    for i in range(img_array.shape[0]):
        print('img_shape : {}'.format(img_array[i].shape))
        print('ori_height : {}, ori_width : {}'.format( img_array[i].shape[0], img_array[i].shape[1]))
        height_resize = img_array[i].shape[0] % 4
        width_resize = img_array[i].shape[1] % 4
        print('img height : {}, img width : {}'.format(height_resize, width_resize))
        if height_resize != 0 or width_resize != 0:
            width = img_array[i].shape[1] - (4 - width_resize)
            height = img_array[i].shape[0] - (4 - height_resize)
            print('width : {}, height : {}'.format(width, height))
            img_array[i] = cv2.resize(img_array[i], dsize=(width, height))

    return img_array

def img_size_check2(img_array):
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
                   

def data_reshape(x_data, y_data):
    x_data = x_data.reshape(1, x_data.shape[0], x_data.shape[1], x_data.shape[2])
    y_data = y_data.reshape(1, y_data.shape[0], y_data.shape[1])

    return x_data, y_data
