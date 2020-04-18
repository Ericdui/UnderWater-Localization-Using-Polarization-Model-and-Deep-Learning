import os
from os import listdir
from os.path import join
import cv2

def main(data_folder_1, data_folder_2):
    folder_dir = join(data_folder_1, data_folder_2)
    folder_names = [name for name in listdir(folder_dir)]

    # exclude '.DS_Store'
    folder_names = sorted(folder_names)[1:]

    for folder_name in folder_names:
        set_dir = join(folder_dir, folder_name)

        image_set_names = [name for name in listdir(set_dir)]
        image_set_names.sort()
        
        for image_set_name in image_set_names:
            # avoid error
            if image_set_name == '.DS_Store':
                continue
                
            image_dir = join(set_dir, image_set_name)
            image = cv2.imread(image_dir)
            
            # down sample
            image = image[::2, ::2]
            image_crop = image[100:1400, 100:1900]
            
            # shape tp 100,100,3
            image_down = image_crop[::13, ::18]
            
            # output 
            output_dir = join(join('239Final_Data', 'Cropped'), folder_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_new_name = 'cropped_' + image_set_name
            cv2.imwrite(join(output_dir, image_new_name), image_down)

if __name__ == '__main__':
    main('239Final_Data', 'Data')