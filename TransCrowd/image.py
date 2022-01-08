from PIL import Image
import numpy as np
import h5py
import cv2

def load_data(img_path, args, train):

    if(train):
        imageDir = 'images_crop'
    else:
        imageDir = 'images'

    img_path = img_path.replace('.h5', '.jpg').replace('gt_density_map', imageDir)
    gt_path = img_path.replace('.jpg', '.h5').replace(imageDir, 'gt_density_map')
    
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            gt_count = np.asarray(gt_file['gt_count'])
            break  # Success!
        except OSError:
            print("load error:", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count
