from __future__ import division

import logging
import math
import os
import platform
if platform.system() == "Linux":
    import resource
import sys
import warnings
from datetime import datetime
from tqdm import trange

import matplotlib.pyplot as plt
import nni
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from nni.utils import merge_parameter
if platform.system() == "Linux":
    from google.colab import drive
    import shutil

import dataset
from config import args, return_args
from image import load_data
from Networks.models import base_patch16_384_gap, base_patch16_384_token
from pytorchtools import EarlyStopping
from utils import save_checkpoint, setup_seed

warnings.filterwarnings('ignore')
import time

listMae = []
listMse = []
epochList = []

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')

def main(args):
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
    elif args['dataset'] == 'VisDrone':
        train_file = './npydata/visDrone_train.npy'
        test_file = './npydata/visDrone_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print("\nLenght Train list: " + str(len(train_list)))
    print("Lenght Test list: " + str(len(val_list)))

    try:
        print('*-----------------------------------------*')
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print('*-----------------------------------------*')
    except:
        pass

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
    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    print(args['best_pred'], args['start_epoch'])
    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(val_list, args, train=False)

    for epoch in range(args['start_epoch'], args['epochs']):

        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        end1 = time.time()

        if epoch % 5 == 0 and epoch >= 10:
            prec1 = validate(test_data, model, args, epoch)
            end2 = time.time()
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])

            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'], F'checkpoint_{epoch}.pth')

            if platform.system() == "Linux":
                try:
                    shutil.copy(F"/content/VisDrone21/TransCrowd" + args['save_path'].replace('.','') + F"/checkpoint_{epoch}.pth", F"/content/gdrive/checkpoint_{epoch}.pth")
                    if (is_best):
                        shutil.copy(F"/content/VisDrone21/TransCrowd" + args['save_path'].replace('.','') + F"/model_best_{epoch}.pth", F"/content/gdrive/model_best_{epoch}.pth")
                except:
                    print("Could not save file to Drive.")
                    pass


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in trange(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

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
       
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True)
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

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
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()

        out1 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out1, gt_count)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('Epoch[{0}/{1}] -> [{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, args['epochs'], i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()

def validate(Pre_data, model, args, epoch):
    print('begin test')
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

    for i, (fname, img, gt_count) in enumerate(test_loader):

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

        if i % 15 == 0:
            print('\n{fname}:\n- Gt {gt:.2f} - Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    f = open("res.txt", "a")
    f.write("Epoch n." + str(epoch) + ": " +str(mae)+"\n")
    f.write("Epoch n." + str(epoch) + ": " +str(mse)+"\n")
    f.write('------\n')
    f.close()
    
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
    nowDate = datetime.now().strftime("%d_%m_%Y_%H_%M")
    pltTitle = 'MAE-MSE_'+ nowDate +'.png'
    plt.savefig(pltTitle)

    if platform.system() == "Linux":
        try:
            shutil.copy(F"/content/VisDrone21/TransCrowd/{pltTitle}", F"/content/gdrive/{pltTitle}")
            print(F'Uploaded image file in: /content/gdrive/{pltTitle}')
            shutil.copy("/content/VisDrone21/TransCrowd/res.txt", F"/content/gdrive/res_{nowDate}.txt")
            print(F'Uploaded result file in: /content/gdrive/res_{nowDate}.txt')
        except:
            print("Could not save file to Drive.")
            pass
    
    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

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
   
    if platform.system() == "Linux":
        print("----------------------------")
        print("** Google Drive Sign In **")
        drive.mount('/content/gdrive')
        print("** Successfully logged in! **")
        print("----------------------------")
    
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
