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

def process_data(data_folder_1, data_folder_2):
    folder_dir = join(data_folder_1, data_folder_2)
    folder_names = [name for name in listdir(folder_dir)]

    # exclude '.DS_Store'
    folder_names = sorted(folder_names)[1:]
    
    train_X = np.zeros((1, 100, 100, 4))
    train_Y = np.zeros((1, 1))
    
    test_X = np.zeros((1, 100, 100, 4))
    test_Y = np.zeros((1, 1))
    
    for folder_name in folder_names:
        set_dir = join(folder_dir, folder_name)

        image_set_names = [name for name in listdir(set_dir)]
        image_set_names.sort()
        
        # get parameters from folder_name
        _, sun_zen, sun_az, cam_head = folder_name.split('_')
        sun_zen = str2int(sun_zen)
        # true environment is 5 not 0
        if sun_zen == 0:
            sun_zen += 5
            
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

            if count <= len(image_set_names) * 0.8:
                train_X = np.concatenate((train_X, data_input.reshape(1, data_input.shape[0], data_input.shape[1], data_input.shape[2])), axis = 0)
                train_Y = np.concatenate((train_Y, cur_aop.reshape(1, 1)), axis = 0)
            else:
                test_X = np.concatenate((test_X, data_input.reshape(1, data_input.shape[0], data_input.shape[1], data_input.shape[2])), axis = 0)
                test_Y = np.concatenate((test_Y, cur_aop.reshape(1, 1)), axis = 0)
            
            count += 1
            
    train_X = train_X.astype('float32')[1:]
    test_X = test_X.astype('float32')[1:]
    train_X = train_X / 255.
    test_X = test_X / 255.
    
    train_Y = train_Y[1:]
    test_Y = test_Y[1:]
    
    return train_X, train_Y, test_X, test_Y