import os
from os import listdir
from os.path import join
import cv2
import numpy as np
from single_scattering import *

def str2int(string):
    if string[0] == 'N':
        convert_int = int(string[1:]) * -1
    else:
        convert_int = int(string)
    
    return convert_int

def process_data(data_folder_1, data_folder_2, ratio, group):
    folder_dir = join(data_folder_1, data_folder_2)
    folder_names = [name for name in listdir(folder_dir)]

    # exclude '.DS_Store'
    folder_names = sorted(folder_names)[1:]
    
    set_dir = join(folder_dir, folder_names[0])
    image_count = len([name for name in listdir(set_dir)])
    
    training_count = round(image_count * ratio)
    
    train_X = np.zeros((training_count * group , 100, 100, 16))
    train_Y = np.zeros((training_count * group, 4))
    
    test_X = np.zeros(((image_count - training_count) * group , 100, 100, 16))
    test_Y = np.zeros(((image_count - training_count) * group  , 4))
    
    # folder index
    folder_idx = 0
        
    for folder_name in folder_names:
        set_dir = join(folder_dir, folder_name)

        image_set_names = [name for name in listdir(set_dir)]
        image_set_names.sort()
        
        # get parameters from folder_name
        set_index, sun_zen, sun_az, cam_head = folder_name.split('_')
        set_index = str2int(set_index)         
        sun_zen = str2int(sun_zen)            
        sun_az = str2int(sun_az)
        cam_head = str2int(cam_head)
        
        cur_aop = oceanaop(sun_az,sun_zen,cam_head,cam_elev=0,m2=1.33,npart=1.08,mu=3.483) * 360 / np.pi
        # keep 2 numbers after pointer
        cur_aop = round(cur_aop, 2)
        
        # to differentiate test and train data
        count = 0
        
        for image_set_name in image_set_names:
            image_dir = join(set_dir, image_set_name)
            image = cv2.imread(image_dir)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            gray_image = gray_image.reshape(gray_image.shape[0], gray_image.shape[1], 1)
        
            data_input = np.concatenate((image, gray_image), axis = 2)
            
            if count < training_count:
                dim_3_low = (folder_idx % 4) * 4
                dim_3_high = (1 + folder_idx % 4) * 4
                dim_0_index = training_count * (set_index - 1) + count
                 
                train_X[dim_0_index, :, :, dim_3_low : dim_3_high] = data_input
                train_Y[dim_0_index, folder_idx % 4] = cur_aop
        
                
            else:
                dim_3_low = (folder_idx % 4) * 4
                dim_3_high = (1 + folder_idx % 4) * 4
                dim_0_index = (len(image_set_names) - training_count) * (set_index - 1) + count - training_count

                test_X[dim_0_index, :, :, dim_3_low : dim_3_high] = data_input
                test_Y[dim_0_index, folder_idx % 4] = cur_aop

            count += 1
            
        folder_idx += 1
            
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.
    test_X = test_X / 255.
    
    train_Y = train_Y
    test_Y = test_Y
    
    return train_X, train_Y, test_X, test_Y