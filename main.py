# -*- coding:utf-8 -*-
import random
import argparse
import os
import warnings
from datetime import datetime
import torch
from torch.backends import cudnn
from network.unet_cm import *
from network.ProbCM import *
from network.unet import *
from trainer import *
from dataset import *
from test import test_model
warnings.filterwarnings("ignore")


def main(configs):

    # load dataset
    centers = ['c1','c2','op_er','cl_di']
    train_loaders = {}
    vali_loaders = {}

    for center in centers:
        train, vali = train_val_split(configs.dataroot+center)
        train_loaders[center] = getTrainLoader(train,configs)
        vali_loaders[center] = getValiLoader(vali,configs)

    imgs = np.load(configs.dataroot+'test/imgs.npy', allow_pickle=True)
    labs = np.load(configs.dataroot+'test/labs.npy', allow_pickle=True)
    test_loaders = getTestLoader({'imgs':imgs,'labs':labs})

    if configs.mode == 'feddan':
        net = ProbCMNet(latent_dim=8,
        in_channels=1,
        num_classes=2,
        low_rank=False,
        num_1x1_convs=3,
        init_features=32,
        lq=configs.lq)
    elif configs.mode == 'fedcm':
        net = UNet_CMs(1,32,4,2,low_rank=False)
    else:
        net = ResUnet(configs,num_cls=2)

    cur_path = os.path.abspath(os.curdir)
    date_time = current_time()
    SAVE_DIR =  cur_path + '/result/' + configs.mode + '_' + configs.loss+date_time +'/'


    # training
    if configs.mode == 'fedavg':
        trainer = Trainer_FedAvg(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    if configs.mode == 'feddan':
        trainer = Trainer_FedDAN(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    if configs.mode == 'fedcm':
        trainer = Trainer_FedCM(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    if configs.mode == 'single':
        trainer = Trainer_Single(net, train_loaders, vali_loaders, configs, SAVE_DIR, 'cl_di')
    elif configs.mode == 'ditto':
        trainer = Trainer_Ditto(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    elif configs.mode == 'fedrep':
        trainer = Trainer_FedRep(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    elif configs.mode == 'fedprox':
        trainer = Trainer_FedProx(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    elif configs.mode == 'fedcurv':
        trainer = Trainer_FedCurv(net, train_loaders, vali_loaders, configs, SAVE_DIR)


    model = trainer.train()

    # save dir
    print(trainer.save_epoch)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    torch.save(model,SAVE_DIR+'model.pth')
    for client, local_model in trainer.models.items():
        torch.save(local_model,SAVE_DIR+'model_{}.pth'.format(client))

    # testing
    test_model(model, test_loaders, SAVE_DIR)

    print('Train finished: ', current_time())

def current_time():
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y, %H:%M:%S")
    return date_time

if __name__ == '__main__':

    # set parameters
    parser = argparse.ArgumentParser()

        # dataset param
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

        # network param
    parser.add_argument('--name', type=str, default='resnet18')
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=256, help='backbone feature')

        # train param
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lq', type=float, default=0.7)
    parser.add_argument('--weight', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--fine_tune', type=bool, default=False)
    parser.add_argument('--fine_tune_epoch', type=int, default=50)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--mode', type=str, default='feddan')
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--cuda', type=int, default=3)

    CONFIGs = parser.parse_args()
    torch.cuda.set_device(CONFIGs.cuda)

    #os.environ["CUDA_VISIBLE_DEVICES"] = CONFIGs.cuda
    cudnn.benchmark = True

    print(CONFIGs)
    main(CONFIGs)
