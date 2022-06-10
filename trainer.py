# -*- coding:utf-8 -*-


import torch
import logging, os
from itertools import cycle
from  torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict
from evaluation import *
import torch.nn.functional as F
import numpy as np
from losses import *
import copy

class Poly(_LRScheduler):

    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=10, last_epoch=-1):

        self.iters_per_epoch = iters_per_epoch

        self.cur_iter = 0

        self.N = num_epochs * iters_per_epoch

        self.warmup_iters = warmup_epochs * iters_per_epoch

        super(Poly, self).__init__(optimizer, last_epoch)



    def get_lr(self):

        T = self.last_epoch * self.iters_per_epoch + self.cur_iter

        factor =  pow((1 - 1.0 * T / self.N), 0.9)

        if self.warmup_iters > 0 and T < self.warmup_iters:

            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch

        self.cur_iter += 1

        assert factor >= 0, 'error in lr_scheduler'

        return [base_lr * factor for base_lr in self.base_lrs]


class Trainer_FedAvg(object):
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):

        # models
        self.models = {client:copy.deepcopy(model).cuda() \
                       for client, data in data_loader_train.items()}

        self.model_cache = copy.deepcopy(model).cuda()

        self.best_model = None

        self.save_epoch = 0

        self.client_weights = {c: torch.tensor(1/len(data_loader_train)).cuda() \
                               for c, data in data_loader_train.items()}

        # train
        self.lr = config.lr
        self.epochs = config.epochs
        self.local_epoch = config.local_epoch
        self.num_data = config.num_data
        self.iters_per_epoch = self.num_data // config.batch_size

        self.criterion = get_loss(config.loss).cuda()

        self.train_loaders = data_loader_train
        self.vali_loaders = data_loader_vali
        self.optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.models.items()}

        # evaluate
        self.epochs_per_vali = 10
        self.evaluator = Evaluator(data_loader_vali)
        # save result
        self.save_dir = save_dir


    def train(self):

        max_dice = 0

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()

            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                meanDice = 0.
                for client in self.models.keys():
                    dice, std = self.evaluator.eval(self.model_cache,client)
                    print(epoch, client, dice, std)
                    meanDice += dice

                if meanDice >= max_dice:
                    max_dice = meanDice
                    self.save_epoch = epoch
                    self.best_model = copy.deepcopy(self.model_cache)

        return self.best_model


    def communication(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                if 'num_batches_tracked' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)


    def train_local(self,client,epoch):

        self.models[client].train()

        for train_batch in self.train_loaders[client]:

            #self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()
            #train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            loss = self.criterion(output, labs)
            loss.backward()
            self.optimizers[client].step()


class Trainer_Naive(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.best_weights = {}

    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.models.items():
                    meanDice,std = self.evaluator.eval(model,client)
                    print(epoch, client, meanDice, std)
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_weights[client] = copy.deepcopy(model.state_dict())

        for client, model in self.models.items():
            model.load_state_dict(self.best_weights[client])

        print(max_dice)
        return self.model_cache


class Trainer_Single(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,center):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.center = center

    def train(self):

        max_dice = 0

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            self.train_local(self.center,epoch)

            lr_scheduler[self.center].step(epoch=epoch)

            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                dice,std = self.evaluator.eval(self.models[self.center],self.center)
                print(epoch, self.center, dice,std)

                if dice >= max_dice:
                    max_dice = dice
                    self.save_epoch = epoch
                    self.best_model = copy.deepcopy(self.models[self.center])

        return self.best_model


class Trainer_FedDAN(Trainer_Naive):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.alpha = config.weight
        self.beta = 0.01
        self.evaluator = Evaluator_feddan(data_loader_vali)

    def train_local(self,client,epoch):

        self.models[client].train()

        for train_batch in self.train_loaders[client]:

            #self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()
            #train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)

            if epoch <50:
                pred,cms = self.models[client](imgs,labs)
                loss = self.models[client].elbo1(pred, cms, labs[:,0],1,self.beta)
            else:
                pred,cms = self.models[client](imgs,labs)
                loss = self.models[client].elbo(pred, cms, labs[:,0],self.alpha,self.beta)

            loss.backward()
            self.optimizers[client].step()

        print(loss)

    def communication(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                #if 'num_batches_tracked' not in key:
                if 'num_batches_tracked' not in key and 'decoders_noisy_layers' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)


############################# Compared Methods #############################################


class Trainer_FedRep(Trainer_Naive):

    def communication(self):

        with torch.no_grad():
            for key in self.model_cache.state_dict().keys():
                if 'num_batches_tracked' not in key and 'decoder.seg_head' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)



class Trainer_Ditto(Trainer_Naive):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.weight = config.weight

        self.local_model = {client:copy.deepcopy(model).cuda() \
                            for client, data in data_loader_train.items()}

        self.local_optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.local_model.items()}


    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}
        local_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.local_optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()

            for client in self.train_loaders.keys():
                self.local_distillation(client)
                local_scheduler[client].step(epoch=epoch)

            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.local_model.items():
                    meanDice,std = self.evaluator.eval(model,client)
                    print(epoch, client, meanDice, std)
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_weights[client] = copy.deepcopy(model.state_dict())

        for client, model in self.models.items():
            model.load_state_dict(self.best_weights[client])

        return self.best_model

    def local_distillation(self, client):

        self.local_model[client].train()

        for train_batch in self.train_loaders[client]:
            self.local_optimizers[client].zero_grad()
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.local_model[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            loss = self.criterion(output, labs)
            loss.backward()
            for p_local, p_global in zip(self.local_model[client].parameters(),self.model_cache.parameters()):
                if p_local.grad is not None:
                    diff = p_local.data-p_global.data
                    p_local.grad += self.weight * diff
            self.local_optimizers[client].step()


class Trainer_FedProx(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.weight = config.weight
        self.omega = defaultdict(lambda: 1)


    def train_local(self,client,epoch):

        self.models[client].train()
        for train_batch in self.train_loaders[client]:

            self.optimizers[client].zero_grad()
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            loss = self.criterion(output, labs)

            distill_loss = self.weight * L2_penalty(self.models[client], self.model_cache, self.omega)

            loss += distill_loss

            loss.backward()
            self.optimizers[client].step()


class Trainer_FedCurv(Trainer_FedProx):

    def train(self):

        max_dice = 0

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)
                    self.diag_fisher(client)
                    lr_scheduler[client].step(epoch=epoch)

            self.communication()


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                meanDice = 0.
                for client in self.models.keys():
                    dice, std = self.evaluator.eval(self.model_cache,client)
                    print(epoch, client, dice, std)
                    meanDice += dice

                if meanDice >= max_dice:
                    max_dice = meanDice
                    self.save_epoch = epoch
                    self.best_model = copy.deepcopy(self.model_cache)

        return self.best_model

    def diag_fisher(self, client):

        precision_matrices = {n: torch.zeros_like(p, dtype=torch.float32).cuda() \
                              for n, p in self.models[client].named_parameters() if p.requires_grad}


        for train_batch in self.train_loaders[client]:
            self.model_cache.train()
            self.model_cache.zero_grad()

            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.model_cache(imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            loss = self.criterion(output, labs)

            loss.backward()

            for n, p in self.models[client].named_parameters():
                if p.grad is not None:
                    precision_matrices[n] += p.grad.data ** 2 / (self.num_data)

        self.omega = {n: p for n, p in precision_matrices.items()}
        self.model_cache.zero_grad()


class Trainer_FedCM(Trainer_FedAvg):
# Input image x instead of feature z in adaptation network 
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.criterion = get_loss('cm_loss',low_rank=False)
        self.seg_loss = get_loss('CE')
        self.lq_loss = get_loss('lq')
        self.alpha = config.weight

    def train_local(self,client,epoch):

        self.models[client].train()

        for train_batch in self.train_loaders[client]:

            #self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()
            #train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            pred,cms = self.models[client](imgs)
            if len(labs.shape) == 4:
                labs = labs[:,0]

            if epoch < 100:
                _,_,TR = self.criterion(pred, cms, [labs], 1)
                loss = self.seg_loss(pred,labs) - TR
            else:
                loss,_,_ = self.criterion(pred, cms, [labs], self.alpha)
                loss = loss + self.lq_loss(pred,labs)

            loss.backward()
            self.optimizers[client].step()

        #print(loss.item(),CE.item(),TR.item())
        print(loss.item())

    def communication(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                #if 'num_batches_tracked' not in key:
                if 'num_batches_tracked' not in key and 'decoders_noisy_layers' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)
