import os
import platform
import random
import re

import h5py
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from PIL import Image
from torchvision import transforms as transformsTorch
from tqdm import trange

cfg_data = EasyDict()

cfg_data.FILE_EXTENSION = ".jpg"
cfg_data.GT_FILE_EXTENSION = ".h5"
cfg_data.LOG_PARA = 2550.0

MEAN = [0.43476477, 0.44504763, 0.43252817]
STD = [0.20490805, 0.19712372, 0.20312176]

class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img = self.lines[index]['img']
        gt_count = self.lines[index]['gt_count']

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
        gt_count = gt_count.copy()
        img = img.copy()

        if self.train == True:
            if self.transform is not None:
                img = self.transform(img)

            return img, gt_count

        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, gt_count

def make_dataframe(folder, train):
    folders = os.listdir(folder)
    dataset = []
    for cur_folder in folders:
        if(cur_folder != 'MC'):
            continue;
        files = os.listdir(os.path.join(folder, cur_folder))
        for file in files:
            if cfg_data.GT_FILE_EXTENSION in file:
                idx = file.split(".")[0]
                if(train):
                    path = folder.replace('train_data','sequences')
                else:
                    path = folder.replace('test_data','sequences')

                dirName = file.split("_")[0]
                imgName = file.split("_")[1].replace("h5","jpg")
                gt = os.path.join(folder, cur_folder, idx + re.sub(', |\(|\)|\[|\]', '_', '') + cfg_data.GT_FILE_EXTENSION)
                if platform.system() == "Linux":
                    dataset.append([idx, os.path.join(path, dirName, imgName), gt])
                else:
                    dataset.append([idx, os.path.join(path, dirName, imgName).replace('/','\\'), gt.replace('/','\\')])

    return pd.DataFrame(dataset, columns=["id", "filename", "gt_filename"])

def pre_data(dataFramePD, train):
    if(train):
        print("Pre loading training dataset ......")
    else:
        print("Pre loading testing dataset ......")
    data_keys = {}
    count = 0
    labelGT = 'density'
    
    for j in trange(len(dataFramePD)):
        Img_path = dataFramePD.loc[j]["filename"]
        GT_path = dataFramePD.loc[j]["gt_filename"]
        fname = os.path.basename(Img_path)

        img = Image.open(Img_path)
        img = img.copy()
                    
        gt_count = np.asarray(h5py.File(GT_path)[labelGT])
        gt_count = gt_count.copy()

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        if j > 100:
            break

    return data_keys

def load_train_val():
    train_df = make_dataframe('../VisDrone2020-CC/train_data', True)
    valid_df = make_dataframe('../VisDrone2020-CC/test_data', False)

    train_data = pre_data(train_df, train=True)
    test_data = pre_data(valid_df, train=False)
    
    train_loader = torch.utils.data.DataLoader(
    VisDroneDataset(train_data,
                        shuffle=True,
                        transform=transformsTorch.Compose
                        ([
                                transformsTorch.ToTensor(),
                                transformsTorch.Normalize(mean=MEAN,std=STD),
                        ]),
                        train=True,
                        batch_size=4,
                        num_workers=2),
    batch_size=4, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
    VisDroneDataset(test_data,
                        shuffle=True,
                        transform=transformsTorch.Compose([
                            transformsTorch.ToTensor(),

                            transformsTorch.Normalize(mean=MEAN,
                                                 std=STD),
                        ]),
                        train=True,
                        batch_size=4,
                        num_workers=2),
    batch_size=4, drop_last=False)

    return train_loader, test_loader
