from PIL import Image
import numpy as np
import h5py
import os
import cv2

def load_data(gt_path, args):

    # Train dataset
    if(os.path.isfile(gt_path.replace('.h5', '.jpg').replace('gt_density_map', "images_crop"))):
        img_path = gt_path.replace('.h5', '.jpg').replace('gt_density_map', "images_crop")
    # Validation dataset
    elif(os.path.isfile(gt_path.replace('.h5', '.jpg').replace('gt_density_map', "images"))):
        img_path = gt_path.replace('.h5', '.jpg').replace('gt_density_map', "images")
    else:
        print(F"Error while opening img path of {gt_path}")
        print("Please check your GT / .JPG path first. Exiting ...")
        exit(0)

    #gt_path = img_path.replace('.jpg', '.h5').replace(imageDir, 'gt_density_map')
    
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            gt_count = np.asarray(gt_file['gt_count'])
            break  # Success!
        except OSError:
            print(F"Load error: {img_path}")
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count
