from easydict import EasyDict as edict
import time

# init
__C = edict()
cfg = __C
now = time.strftime("%m-%d_%H-%M", time.localtime())

#------------------------------TRAIN------------------------
__C.SEED = 3035 # random seed,  for reporduction
__C.DATASET = 'VIS' # dataset selection
__C.NET = 'MobileCount' # net selection: MobileCount, MobileCountx1_25, MobileCountx2

__C.TRAIN_BATCH_SIZE = 1
__C.N_WORKERS = 2

__C.PRE_TRAINED = None

__C.DEVICE = 'cuda'  # cpu or cuda
__C.GPU_ID = [0] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-4 # learning rate
__C.LR_DECAY = 0.995 # decay rate
__C.LR_DECAY_START = -1 # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1 # decay frequency

# Epochs
__C.INIT_EPOCH = 0
__C.MAX_EPOCH = 60

# Save directory
__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes
__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

# print 
__C.PRINT_FREQ = 10

#------------------------------VAL------------------------
__C.VAL_SIZE = 0.2
__C.VAL_BATCH_SIZE = 1
__C.VAL_DENSE_START = 1
__C.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes

#================================================================================
#================================================================================
#================================================================================  
