from __future__ import division

import logging
import math

import os
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import platform
import warnings

if platform.system() == "Linux":
    import shutil 

import nni
import numpy as np
import torch
import torch.nn as nn
from nni.utils import merge_parameter
from torchvision import transforms
from tqdm import tqdm, trange

import dataset
from config import args, return_args
from image import load_data
from Networks.models import base_patch16_384_gap, base_patch16_384_token
from utils import setup_seed

warnings.filterwarnings('ignore')

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):

    # Args for debugging through IDE
    #args['dataset'] = 'VisDrone'                   # Replace with you own dataset
    #args['save_path'] = 'save_file/VisDrone'       # Directory where to save models
    #args['uses_drive'] = False                     # Whether to choose Drive to save models
    #args['model_type'] = 'gap'                     # Choose your model type (Token) / (Gap)
    #args['loadModel'] = 'model_best_Group_B.pth'   # Load weights from previous model

    # We set workers according to the warning about max threads 
    # by NumExpr and set it to 8 - (i5-10600 @3.30 GHz).
    # Change accordingly to your CPU at line 6.
    args['workers'] = int(os.environ['NUMEXPR_NUM_THREADS'])

    # Print the arguments
    print("Starting validation with the following configuration:")
    print("Batch size: {}".format(args['batch_size']))
    print("Dataset: {}".format(args['dataset']))
    print("Load model: {}".format(args['loadModel']))
    print("Model type: {}".format(args['model_type']))
    print("Save path: {}".format(args['save_path']))
    print("Uses Drive: {}".format(args["uses_drive"]))
    print("Workers: {}".format(args['workers']))
    print("---------------------------------------------------")

    if platform.system() == "Linux" and args['uses_drive']:
        print("----------------------------")
        print("** Google Drive Sign In **")
        if not(os.path.exists("../../gdrive/")):
            print("No Google Drive path detected! Please mount it before running this script or disable ""uses_drive"" flag!")
            print("----------------------------")
            exit(0)
        else:
            print("** Successfully logged in! **")
            print("----------------------------")

    if args['dataset'] == 'ShanghaiA':
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        test_file = './npydata/nwpu_val.npy'
    elif args['dataset'] == 'VisDrone':
        test_file = './npydata/visDrone_test.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print("Lenght Test list: " + str(len(val_list)))

    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print("===================================================")
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
        print("===================================================")
    else:
        device = torch.device("cpu")
        print('Cuda not available. Using CPU.')

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    if args['model_type'] == "token":
        model = base_patch16_384_token(pretrained=True)
    else:
        model = base_patch16_384_gap(pretrained=True)
    print("===================================================")

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    print("Save path: {}".format(args['save_path']))
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if(args['uses_drive']):
        if not os.path.exists(F"../../gdrive/MyDrive/VisDroneResults/{args['save_path']}"):
            os.makedirs(F"../../gdrive/MyDrive/VisDroneResults/{args['save_path']}")

    if args['loadModel']:
        print("You specified a pre-loading directory for a model.")
        print("The directory is: {}".format(args["loadModel"]))
        if os.path.isfile(args["loadModel"]):
            print("=> Loading model '{}'".format(args["loadModel"]))
            modelLoaded = torch.load(args['loadModel'], device)
            model.load_state_dict(modelLoaded['state_dict'], strict=False)
            # Best precision
            try:
                bestPrecision = modelLoaded['best_prec1']
                print("- Best precision from model: {}".format(bestPrecision))
            except:
                bestPrecision = 0
                print("- No best precision present in this model")
                pass
            print("Custom model loaded successfully")
        else:
            print("=> No model found at '{}'".format(args["loadModel"]))
            print("Are you sure the directory / model exist? Exiting..")
            exit(0)
        print("===================================================")

    print(F"Setting {args['workers']} threads for Torch...")
    torch.set_num_threads(args['workers'])
    print("Successfully set threads.")

    test_data = pre_data(val_list, args, train=False)
    print("===================================================")

    # Inference
    prec1 = validate(test_data, model, args)

    maeModel = '- Best MAE from model: {mae:.9f} '.format(mae=bestPrecision)
    maeValidation = '- Best MAE from validation: {mae:.9f} '.format(mae=prec1)

    print("===================================================")
    print("Validation ended successfully.")
    print(maeModel)
    print(maeValidation)
    print("===================================================")

    f = open(args['save_path'] + "/res.txt", "a")
    f.write(maeModel)
    f.write("\n===================================================\n")
    f.write(maeValidation)
    f.close()

    if platform.system() == "Linux" and args['uses_drive']:
        try:
            shutil.copy(args['save_path'] + "/res.txt", F"../../gdrive/MyDrive/VisDroneResults/{args['save_path']}/res.txt")
            print(F"Uploaded result file in: /content/gdrive/MyDrive/VisDroneResults/{args['save_path']}/res.txt")
        except:
            print("Could not save file to Drive.")
            pass


def pre_data(val_list, args, train):
    print("Pre-loading dataset ......")
    data_keys = {}
    count = 0
    for j in trange(len(val_list)):
        Img_path = val_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys

def validate(test_data, model, args):
    print("\nStarting validation...")

    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=batch_size)

    batch_bar = tqdm(total=len(test_loader), desc="Batch", position=0)

    model.eval()

    mae = 0.0
    mse = 0.0

    for idx, (fname, img, gt_count) in tqdm(enumerate(test_loader)):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
                
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        batch_bar.update(1)

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
