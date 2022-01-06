import numpy as np
import pandas as pd
import os
import re
import torch
import torchvision
from PIL import Image as pil
import h5py
from config import cfg as config
from easydict import EasyDict
import sklearn.model_selection
import misc.transforms as transforms
import cv2

cfg_data = EasyDict()

cfg_data.FILE_EXTENSION = ".jpg"
cfg_data.GT_FILE_EXTENSION = ".h5"
cfg_data.LOG_PARA = 2550.0

MEAN = [0.43476477, 0.44504763, 0.43252817]
STD = [0.20490805, 0.19712372, 0.20312176]

class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, train=True, img_transform=None, gt_transform=None):
        self.dataframe = dataframe
        self.train = train
        self.train_transforms = None
        self.img_transform = None
        self.gt_transform = None
        if train:
            self.train_transforms = transforms.RandomHorizontallyFlip()

        if img_transform:
            # Initialize data transforms
            self.img_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=MEAN, std=STD),
                ]
            )  # normalize to (-1, 1)
        if gt_transform:
            self.gt_transform = torchvision.transforms.Compose(
                [transforms.Scale(cfg_data.LOG_PARA)]
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        # Obtain the filename and target
        filename = self.dataframe.loc[i]["filename"]
        target_filename = self.dataframe.loc[i]["gt_filename"]

        # Load the img and the ground truth
        with pil.open(filename).convert('RGB') as img:
            data = np.array(img, dtype=np.uint8)

        with h5py.File(target_filename, 'r') as hf:
            target = np.array(hf.get("gt_count"), dtype=np.uint8)
        hf.close()

        if(self.train == True):
            return data, target

        else:
            if self.train_transforms:
                data, target = self.train_transforms(data, target)

            if self.img_transform:
                data = self.img_transform(data)

            if self.gt_transform:
                target = self.gt_transform(target)

            width, height = data.shape[2], data.shape[1]

            m = int(width / 384)
            n = int(height / 384)
            for i in range(0, m):
                for j in range(0, n):

                    if i == 0 and j == 0:
                        img_return = data[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                        target_return = target[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                    else:
                        crop_img = data[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                        crop_target = target[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)

                        img_return = torch.cat([img_return, crop_img], 0).cuda()
                        target_return = torch.cat([target_return, crop_target], 0).cuda()

            return img_return, target_return

    def get_targets(self):
        return self.targets


def make_dataframe(folder):
    # Return a DataFrame with columns (example folder, example idx, filename, gt filename)
    folders = os.listdir(folder)
    dataset = []
    for cur_folder in folders:
        files = os.listdir(os.path.join(folder, cur_folder))
        for file in files:
            if cfg_data.FILE_EXTENSION in file:
                idx = file.split(".")[0]
                gt = os.path.join(folder, cur_folder.replace('images_crop', 'gt_density_map'), idx + re.sub(', |\(|\)|\[|\]', '_', '') + cfg_data.GT_FILE_EXTENSION)
                dataset.append([idx, os.path.join(folder, cur_folder, file).replace('/','\\'), gt.replace('/','\\')])

    return pd.DataFrame(dataset, columns=["id", "filename", "gt_filename"])


def load_test():
    df = make_dataframe("../VisDrone2020-CC/test_data")
    ds = VisDroneDataset(df, train=False, gt_transform=False)
    return ds


def load_train_val():
    train_df = make_dataframe('../VisDrone2020-CC/train_data')
    valid_df = make_dataframe('../VisDrone2020-CC/test_data')

    train_set = VisDroneDataset(train_df)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=config.N_WORKERS,
        shuffle=True,
    )

    val_set = VisDroneDataset(valid_df)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config.VAL_BATCH_SIZE, num_workers=config.N_WORKERS, shuffle=True
    )

    return train_loader, val_loader


def load_test_dataset(data_folder, dataclass, make_dataframe_fun):
    # Load the test dataframe
    test_df = make_dataframe_fun(data_folder)

    # Instantiate the dataset
    test_data = dataclass(data_folder, test_df)
    return test_data
