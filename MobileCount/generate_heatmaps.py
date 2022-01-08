import os

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import platform

GAMMA = 3
FILENAME_LEN = 5

trainDataDirectory = '../VisDrone2020-CC/train_data/MC'
testDataDirectory = '../VisDrone2020-CC/test_data/MC'

if not os.path.exists(trainDataDirectory):
    os.makedirs(trainDataDirectory)
if not os.path.exists(testDataDirectory):
    os.makedirs(testDataDirectory)

gt_paths=[]

for k in range(1,112):
    if k < 10:
        gt_paths.append('../VisDrone2020-CC/annotations' + '/0000' + str(k) + '.txt')
    elif k < 100:
        gt_paths.append('../VisDrone2020-CC/annotations' + '/000' + str(k) + '.txt')
    else:
        gt_paths.append('../VisDrone2020-CC/annotations' + '/00' + str(k) + '.txt')
gt_paths.sort()

testList = np.loadtxt('../VisDrone2020-CC/testlist.txt', delimiter=" ", dtype="str") 
trainList = np.loadtxt('../VisDrone2020-CC/trainlist.txt', delimiter=" ", dtype="str")

size = (270, 480)

def generate_heatmap(df, img_size, wished_heatmap_size):
    heatmaps = {}
        
    frames = np.unique(df['frame'].values)
    img_size = np.array(img_size)
    wished_heatmap_size = np.array(wished_heatmap_size)
    ratio = (img_size / wished_heatmap_size)
    for frame in frames:
        heads = np.rint(df[df['frame'] == frame][['x', 'y']].values / ratio).astype('int64')
        heatmap = np.zeros(wished_heatmap_size)
        heatmap[np.clip(heads[:, 1], 0, wished_heatmap_size[0]) - 1,
                np.clip(heads[:, 0] - 1, 0, wished_heatmap_size[1]) - 1] = 1
        heatmap = gaussian_filter(heatmap, GAMMA / ratio)
        heatmaps[frame] = heatmap
    return heatmaps


def make_ground_truth(folder, img_folder, isTrain, dataframe_fun, size=None):
    gt_files = os.listdir(folder)
    
    if(isTrain):
        gt_files = trainList.tolist()
        print("\nGenerating ground truth for training set...")
    else:
        gt_files = testList.tolist()
        print("\nGenerating ground truth for test set...")

    for gt in tqdm(gt_files):
        seq_folder = img_folder + F"\{gt}"

        if platform.system() == "Linux":
          seq_folder = seq_folder.replace('\\','/')

        folderJoined = os.path.join(seq_folder, list(filter(lambda x: '.jpg' in x, os.listdir(seq_folder)))[0])

        img_size = plt.imread(folderJoined).shape[:2]

        if size is None:
            size = img_size
            
        df = dataframe_fun(os.path.join(folder, gt + '.txt'))
        heatmaps = generate_heatmap(df, img_size, size)
        
        for heatmap in heatmaps:
            if heatmap < 10:
                frame_name = '0000' + str(heatmap)
            else:
                frame_name = '000' + str(heatmap)
                
            if platform.system() == "Linux":
              split = seq_folder.split('/')[0] + F"/{seq_folder.split('/')[1]}" + F"/{seq_folder.split('/')[2]}"
              image_name = seq_folder.split('/')[3]
            else:
              split = seq_folder.split('\\')[0]
              image_name = seq_folder.split('\\')[1]
            
            if(isTrain):
                path = os.path.join(split.replace('sequences','train_data/MC'), image_name) + "_" + frame_name
            else:
                path = os.path.join(split.replace('sequences','test_data/MC'), image_name) + "_" + frame_name
                        
            hf = h5py.File(path + ".h5", 'w')
            hf.create_dataset('density', data=heatmaps[heatmap])
            hf.close()
            

def dataframe_load_test(filename):
    df = pd.read_csv(filename, header=None)
    df.columns = ['frame', 'head_id', 'x', 'y', 'width', 'height', 'out', 'occl', 'mistero', 'boh']
    df['x'] = df['x'] + df['width'] // 2
    df['y'] = df['y'] + df['height'] // 2

    df = df[(df['frame'] % 10) == 1]
    df['frame'] = df['frame'] // 10 + 1
    return df[['frame', 'x', 'y']]


def dataframe_load_train(filename):
    df = pd.read_csv(filename, header=None)
    df.columns = ['frame', 'x', 'y']
    return df


def main():
    
    train = [dataframe_load_train, size]
    test = [dataframe_load_test, size]

    make_ground_truth('../VisDrone2020-CC/annotations',
                      '../VisDrone2020-CC/sequences',
                      True,
                      *train)
    
    make_ground_truth('../VisDrone2020-CC/annotations',
                      '../VisDrone2020-CC/sequences',
                      False,
                      *test)