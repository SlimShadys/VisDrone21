import torch
import shutil
import numpy as np
import random
import pandas as pd
            
def save_checkpoint(state,is_best, task_id, filename):
    torch.save(state, './'+str(task_id)+'/'+filename)
    if is_best:
        shutil.copyfile('./'+str(task_id)+'/'+filename, './'+str(task_id)+'/'+F'model_best_{filename.split("_")[1].split(".")[0]}.pth')

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataframe_load_test(filename):
    """
    Load the dataframe for the test set csv format of VisDrone
    @param filename: csv path
    @return: dataframe of columns [frame, x, y]
    """
    df = pd.read_csv(filename, header=None)
    df.columns = ['frame', 'head_id', 'x', 'y', 'width', 'height', 'out', 'occl', 'undefinied', 'undefinied']
    df['x'] = df['x'] + df['width'] // 2
    df['y'] = df['y'] + df['height'] // 2

    df = df[(df['frame'] % 10) == 1]
    df['frame'] = df['frame'] // 10 + 1
    return df[['frame', 'x', 'y']]
