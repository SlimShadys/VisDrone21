from __future__ import division

import logging
import math

import os
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import platform

if platform.system() == "Linux":
    import resource

import sys
import warnings
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import nni
import numpy as np
import torch
import torch.nn as nn
from nni.utils import merge_parameter
from torchvision import transforms
from tqdm import tqdm, trange

if platform.system() == "Linux":
    import shutil

import dataset
from config import args, return_args
from image import load_data
from Networks.models import base_patch16_384_gap, base_patch16_384_token
from pytorchtools import EarlyStopping
from utils import estimatedTime, save_checkpoint, setup_seed

warnings.filterwarnings('ignore')
import datetime

listMae = []
listMse = []
epochList = []

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):
    
    # Args for debugging through IDE
    #args['dataset'] = 'VisDrone20'                # Replace with you own dataset
    #args['save_path'] = './save_file/VisDrone20'  # Directory where to save models
    #args['uses_drive'] = False                  # Whether to choose Drive to save models
    #args['model_type'] = 'gap'                  # Choose your model type (Token) / (Gap)
    #args['batch_size'] = 8                      # Batch size for training
    #args['epochs'] = 50                         # Numbers of epochs
    #args['loadModel'] = './loadModel/model_best_ShanghaiTech_GAP.pth'  # Load weights from previous model

    # We set workers according to the warning about max threads 
    # by NumExpr and set it to 8 - (i5-10600 @3.30 GHz).
    # Change accordingly to your CPU at line 6.
    args['workers'] = int(os.environ['NUMEXPR_NUM_THREADS'])

    # Print the arguments
    print("Starting training with the following configuration:")
    print("Batch size: {}".format(args['batch_size']))
    print("Checkpoint model: {}".format(args['checkpoint']))
    print("Dataset: {}".format(args['dataset']))
    print("Epochs: {}".format(args['epochs']))
    print("Group name: {}".format(args['group_name']))
    print("Learning rate: {}".format(args['lr']))
    print("Load model: {}".format(args['loadModel']))
    print("Model type: {}".format(args['model_type']))
    print("Momentum: {}".format(args['momentum']))
    print("Patience: {}".format(args['patience']))
    print("Save path: {}".format(args['save_path']))
    print("Start epoch: {}".format(args["start_epoch"]))
    print("Uses Drive: {}".format(args["uses_drive"]))
    print("Weight decay: {}".format(args['weight_decay']))
    print("Workers: {}".format(args['workers']))
    
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
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'
    elif args['dataset'] == 'VisDrone20':
        train_file = './npydata/visDrone20_train.npy'
        test_file = './npydata/visDrone20_test.npy'
    elif args['dataset'] == 'VisDrone21':
        train_file = './npydata/visDrone21_train.npy'
        test_file = './npydata/visDrone21_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    print("===================================================")
    print("Lenght Train list: " + str(len(train_list)))
    print("Lenght Test list: " + str(len(test_list)))
    print("===================================================")

    if platform.system() == "Linux" and args['uses_drive']:
        if (args['group_name'] is None):
            groupName = 'A'
            print("You didn't select any group name! Assuming 'A'")
        else:
            groupName = args['group_name']
            print(F"Group name: {groupName}")
        print("===================================================")
    else:
        groupName = ''

    if(torch.cuda.is_available()):
        device = torch.device("cuda")
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

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion = nn.L1Loss(size_average=False).cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(),
             'lr': args['lr']
            },
        ],
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)

    if platform.system() == "Linux" and args['uses_drive']:
        if not os.path.exists(F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/checkpoint/"):
            os.makedirs(F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/checkpoint")

        if not os.path.exists(F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/best/"):
            os.makedirs(F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/best")

    print("===================================================")
    print(F"The directory for saving checkpoints/models is: {args['save_path']}")
    if not(os.path.exists(os.path.join(args['save_path'], 'checkpoint'))):
        os.makedirs(os.path.join(args['save_path'], 'checkpoint'))

    if not(os.path.exists(os.path.join(args['save_path'], 'best'))):
        os.makedirs(os.path.join(args['save_path'], 'best'))
    print("===================================================")

    if args['loadModel']:
        print("You specified a pre-loading directory for a model.")
        print("The directory is: {}".format(args["loadModel"]))
        if os.path.isfile(args["loadModel"]):
            print("=> Loading model '{}'".format(args["loadModel"]))
            modelLoaded = torch.load(args['loadModel'], device)

            # State
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
        
    if args['checkpoint']:
        print("You specified a pre-loading directory for checkpoints.")
        print(F"The directory is: {args['checkpoint']}")
        if os.path.isfile(args['checkpoint']):
            print("=> Loading checkpoint '{}'".format(args['checkpoint']))
            checkpoint = torch.load(args['checkpoint'])

            # State & Optimizer
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])

            # Best precision
            try:
                args['best_pred'] = checkpoint['best_prec1']
            except:
                print("- No best precision present in this model")
                pass

            # Starting epoch
            try:
                args['start_epoch'] = checkpoint['epoch']
            except:
                print("- No best precision present in this model")
                pass
            
            print("Checkpoint loaded successfully")
        else:
            print("=> No checkpoint found at '{}'".format(args['checkpoint']))
            print("Are you sure the directory / checkpoint exist? Exiting..")
            exit(0)
        print("===================================================")

    print(F"Setting {args['workers']} threads for Torch...")
    torch.set_num_threads(args['workers'])
    print("Successfully set threads.")

    print("===================================================")
    print(F"- Best prediction loaded: {args['best_pred']}")
    print(F"- Starting from epoch n. {args['start_epoch']}")
    print("===================================================")

    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(test_list, args, train=False)
    print("===================================================")

    tempoTrascorso = datetime.timedelta(seconds=0)

    for epoch in range(args['start_epoch'], args['epochs']):
            
        start = 0
        end = 0
        i = 0

        start = timer()

        losses = train(train_data, model, criterion, optimizer, epoch, args, scheduler)

        prec1, mse = validate(test_data, model, args, epoch + 1, groupName)

        if(prec1 < args['best_pred']):
            is_best = True
            bestEpoch = epoch + 1
        else:
            is_best = False

        args['best_pred'] = min(prec1, args['best_pred'])

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': args['best_pred'],
        }, is_best, args['save_path'], epoch + 1)

        f = open(args['save_path'] + "/res.txt", "a")

        if(is_best):
            write = 'Epoch[{0}/{1}] || Loss val: {loss.val:.4f} (Average: {loss.avg:.4f}) - MAE: {MAE:.4f} - MSE: {MSE:.4f} || [saved]\n'.format(epoch + 1, args['epochs'], MAE=prec1, MSE=mse, loss=losses)
        else:
            write = 'Epoch[{0}/{1}] || Loss val: {loss.val:.4f} (Average: {loss.avg:.4f}) - MAE: {MAE:.4f} - MSE: {MSE:.4f} ||\n'.format(epoch + 1, args['epochs'], MAE=prec1, MSE=mse, loss=losses)
        print('\n' + write)
        
        f.write(write)
        f.write('=============================================\n')
        f.close()

        print('* Best MAE until now: {mae:.4f} (Epoch n. {epoch})'.format(mae=args['best_pred'], epoch=bestEpoch))

        if platform.system() == "Linux" and args['uses_drive']:
            try:
                if (is_best):
                    shutil.copy(args['save_path'] + F"/best/model_best_epoch-{epoch + 1}.pth", F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/best/model_best_epoch-{epoch + 1}.pth")
                    print(F'Uploaded model_best_epoch-{epoch + 1}.pth in: /content/gdrive/MyDrive/VisDroneResults/Groups_{groupName}/best/model_best_epoch-{epoch + 1}.pth')
                else:
                    shutil.copy(args['save_path'] + F"/checkpoint/checkpoint_epoch-{epoch + 1}.pth", F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/checkpoint/checkpoint_epoch-{epoch + 1}.pth")
                    print(F'Uploaded checkpoint_epoch-{epoch + 1}.pth in: /content/gdrive/MyDrive/VisDroneResults/Groups_{groupName}/checkpoint/checkpoint_epoch-{epoch + 1}.pth')
                shutil.copy(args['save_path'] + "/res.txt", F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/res_{groupName.split('/')[1]}.txt")
                print(F'Uploaded result file in: /content/gdrive/MyDrive/VisDroneResults/Groups_{groupName}/res_{groupName.split("/")[1]}.txt')
            except:
                print("Could not save file to Drive.")
                pass

        end = timer()
        tempoTrascorso = estimatedTime(end-start, i, args["epochs"], epoch+1, tempoTrascorso)
        i += 1
        print("===================================================")

    print("\n### =================================================== ###")
    print("Done training the model.")
    print('Best MAE was: {mae:.3f} '.format(mae=args['best_pred']))

def pre_data(image_list, args, train):
    if(train):
        print("Pre loading training dataset ......")
    else:
        print("Pre loading testing dataset ......")
    data_keys = {}
    count = 0
    for j in trange(len(image_list)):
        Img_path = image_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        #if j > 5:
        #    break

    return data_keys

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):

    losses = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)

    args['lr'] = optimizer.param_groups[0]['lr']

    print(F"Starting Epoch n. {epoch + 1} / {args['epochs']}")
    print(F"\t- {epoch * len(train_loader.dataset)} samples processed until now at a learning rate of {args['lr']}")

    model.train()

    for i, (fname, img, gt_count) in enumerate(tqdm(train_loader)):

        nameFile = fname
        
        img = img.cuda()

        out1 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out1, gt_count)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
           
    scheduler.step()

    return losses

def validate(Pre_data, model, args, epoch, groupName):

    print('Beginning validation on test dataset...')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
                        dataset.listDataset(Pre_data,
                                            args['save_path'],
                                            shuffle=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]
                                                ),
                                            ]),
                                            args=args,
                                            train=False),
                        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0

    for idx, (fname, img, gt_count) in enumerate(tqdm(test_loader)):

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

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    listMae.append(mae)
    listMse.append(mse)

    epochList.append(epoch)
    default_x_ticks = range(len(epochList))

    plt.figure(figsize=(8,8))

    plt.plot(listMae, label='MAE')
    plt.plot(listMse, label='MSE')

    plt.ylabel('MAE/MSE')
    plt.xlabel('Epochs')
    plt.xticks(default_x_ticks, epochList)
    plt.legend(loc='upper right', prop={'size': 15})
    nowDate = datetime.datetime.now().strftime("%H_%M")
    pltTitle = 'MAE-MSE_'+ nowDate + '_' + 'Epoch_' + str(epoch) +'.png'
    plt.savefig(args['save_path'] + "/" + pltTitle)

    if platform.system() == "Linux" and args['uses_drive']:
        try:
            shutil.copy(args['save_path'] + "/" + pltTitle, F"../../gdrive/MyDrive/VisDroneResults/Groups_{groupName}/{pltTitle}")
            print(F'Uploaded image file in: /content/gdrive/MyDrive/VisDroneResults/Groups_{groupName}/{pltTitle}')
        except:
            print("Could not save file to Drive.")
            pass

    return mae, mse

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

def memory_limit(percentage: float):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))
    print(f'My memory is limited to {percentage}%')

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    print("Free memory available: " + str(free_memory))
    return free_memory

def memory(percentage):
    if platform.system() != "Linux":
        print('Limiting memory only works on Linux!')
        return
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 /1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator

if __name__ == '__main__':
    memory(0.95)
   
    print("|***************************************************|")
    print("|                T R A N S C R O W D                |")
    print("|                 Made by: Dk-Liang                 |")
    print("|---------------------------------------------------|")
    print("|              Revamped by: SlimShadys              |")
    print("|         ---  Student at University of  ---        |")
    print("|     --- Computer Science and Technologies ---     |")
    print("|     ---                for                ---     |")
    print("|     ---        Software Production        ---     |")
    print("|         ---        Bary, Italy        ---         |")
    print("|***************************************************|")
   
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
