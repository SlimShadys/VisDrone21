from VIS import load_train_val, cfg_data
from train import Trainer
from models.CC import CrowdCounter
from config import cfg
import sys
import generate_heatmaps

def load_CC():
    cc = CrowdCounter([0], cfg.NET)
    if cfg.PRE_TRAINED:
        cc.load(cfg.PRE_TRAINED)
    return cc

def beginTrain():
    #generate_heatmaps.main()
    trainer = Trainer(dataloader=load_train_val, cfg_data=cfg_data, net_fun=load_CC)
    trainer.train()

def beginTest():
    trainer = Trainer(dataloader=load_train_val, cfg_data=cfg_data, net_fun=load_CC)
    trainer.train()

if __name__ == "__main__":

    # argumentsList = sys.argv[1:]
    
    # if("--train" in argumentsList and "--test" in argumentsList):
    #     print("You can only run training or testing separately. Not both!")
    #     exit(0)
    # elif("--train" in argumentsList):
         beginTrain()
    # elif("--test" in argumentsList):
    #     beginTest()
    # else:
    #     print("Please provide a parameter for testing or training.")
    #     exit(0)
