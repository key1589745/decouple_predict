import torch
from torch import nn
import math
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

def get_loss(name, **kwargs):
    if name == 'CE':
        return nn.CrossEntropyLoss()
    elif name == 'alea':
        return AleaLoss(**kwargs)
    elif name == 'L2':
        return L2Loss(**kwargs)
    elif name == 'kld':
        return kld_loss()
    elif name == 'truncted':
        return TruncatedLoss(**kwargs)
    elif name == 'cm_loss':
        if kwargs['low_rank']:
            return noisy_label_loss_low_rank
        else:
            return noisy_label_loss
    elif name == 'lq':
        return Lq_loss()
    else:
        raise NotImplementedError
        

class L2Loss(nn.Module):

    def __init__(self,one_hot=True):
        super(L2Loss, self).__init__()
        self.one_hot = one_hot
    
    def forward(self, predict, y):
        if self.one_hot:
            y = F.one_hot(y, 2).permute(0,3,1,2)
        
        loss = torch.norm(y-predict,2,1)**2 

        return torch.mean(loss)

def L2_penalty(model, model_cache, omega=defaultdict(lambda: 1)):
    loss = 0
    params = {n: p.data.cuda() for n, p in model_cache.state_dict().items() if p.requires_grad}
    for n, p in model.state_dict().items():
        if p.requires_grad:
            _loss = omega[n] * (p - params[n]) ** 2
            loss += _loss.sum()
    return loss
    
class kld_loss(nn.Module):

    def __init__(self):
        super(kld_loss, self).__init__()

    def forward(self, mu, log_var):

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

        return kld_loss
        
class AleaLoss(nn.Module):

    def __init__(self,one_hot=True):
        super(AleaLoss, self).__init__()
        self.one_hot = one_hot

        
    def forward(self, predict, sigma, y):

        #s = torch.log(sigma)
        if self.one_hot:
            y = F.one_hot(y, 2).permute(0,3,1,2)
        
        loss = 1/2*torch.exp(-sigma) * torch.norm(y-predict,2,1)**2 + 1/2*torch.exp(sigma)

        return torch.mean(loss)
    

class Lq_loss(nn.Module):

    def __init__(self, trainset_size=20, imgsize=256, q=0.7, k=0.8):
        super(Lq_loss, self).__init__()
        self.q = q
        self.k = k
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)

        Yg = torch.gather(p, 1, targets.unsqueeze(1))

        loss = (1-(Yg**self.q))/self.q
        loss = torch.mean(loss)

        return loss

class TruncatedLoss(nn.Module):

    def __init__(self, trainset_size=20, imgsize=256, q=0.7, k=0.8):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1, imgsize, imgsize), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)

        Yg = torch.gather(p, 1, targets)

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, targets)
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = torch.ones_like(Lq) * (1-self.k**self.q)/self.q
        
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)
        

class KL_distill_loss(nn.KLDivLoss):
    
    def __init__(self, size_average=None, reduce=None, reduction = 'mean', log_target = True):
        super().__init__(size_average, reduce, reduction, log_target)

    def forward(self, inputs, targets):
        
        inputs = F.log_softmax(inputs, 1)
        targets = F.log_softmax(targets, 1)

        return super().forward(inputs, targets)    
    
def L2_penalty(model, model_cache, omega=defaultdict(lambda: 1)):
    loss = 0
    params = {n: p.data.cuda() for n, p in model_cache.state_dict().items() if p.requires_grad}
    for n, p in model.state_dict().items():
        if p.requires_grad:
            _loss = omega[n] * (p - params[n]) ** 2
            loss += _loss.sum()
    return loss


def feature_distill_loss(old_stats,new_stats):
    
    #forcing mean and variance to match between two distributions
    #other ways might work better, i.g. KL divergence
    r_feature = 0.
    for old, new in zip(old_stats, new_stats):
        #print(new.stats['var'].requires_grad)
        r_feature += torch.norm(old['var'] - new[0], 2) + \
        torch.norm(old['mean'] - new[1], 2)
    
    return r_feature


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce = self.bce_loss(input, target)
        pt = torch.exp(-bce)
        if self.alpha is None:
            # use inverse weighting
            zeros = (1 - target).sum()
            total = target.numel()
            alpha = zeros / total
        else:
            # use fixed weighting
            alpha = self.alpha

        focal_loss = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()

def iou_loss(pred, target):
    y_pred = torch.sigmoid(pred.view(-1))
    y_target = target.view(-1)
    intersection = (y_pred * y_target).sum()
    iou = (intersection + 1) / (y_pred.sum() + y_target.sum() - intersection + 1)

    return -iou 


def noisy_label_loss(pred, cms, labels, alpha=0.1):
    """ This function defines the proposed trace regularised loss function, suitable for either binary
    or multi-class segmentation task. Essentially, each pixel has a confusion matrix.
    Args:
        pred (torch.tensor): output tensor of the last layer of the segmentation network without Sigmoid or Softmax
        cms (list): a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        labels (torch.tensor): labels
        alpha (double): a hyper-parameter to decide the strength of regularisation
    Returns:
        loss (double): total loss value, sum between main_loss and regularisation
        main_loss (double): main segmentation loss
        regularisation (double): regularisation loss
    """
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()

    # normalise the segmentation output tensor along dimension 1
    pred_norm = nn.Softmax(dim=1)(pred)

    # b x c x h x w ---> b*h*w x c x 1
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    for cm, label_noisy in zip(cms, labels):
        # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
        # label_noisy: noisy label, b x h x w

        # b x c**2 x h x w ---> b*h*w x c x c
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # normalisation along the rows:
        cm = cm / cm.sum(1, keepdim=True)

        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)

    regularisation = alpha*regularisation
    loss = main_loss + regularisation

    return loss, main_loss, regularisation


def noisy_label_loss_low_rank(pred, cms, labels, alpha):
    """ This function defines the proposed low-rank trace regularised loss function, suitable for either binary
    or multi-class segmentation task. Essentially, each pixel has a confusion matrix.
    Args:
        pred (torch.tensor): output tensor of the last layer of the segmentation network without Sigmoid or Softmax
        cms (list): a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        labels (torch.tensor): labels
        alpha (double): a hyper-parameter to decide the strength of regularisation
    Returns:
        loss (double): total loss value, sum between main_loss and regularisation
        main_loss (double): main segmentation loss
        regularisation (double): regularisation loss
    """

    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    # pred: b x c x h x w
    pred_norm = nn.Softmax(dim=1)(pred)
    # pred_norm: b x c x h x w
    pred_norm = pred_norm.view(b, c, h*w)
    # pred_norm: b x c x h*w
    pred_norm = pred_norm.permute(0, 2, 1).contiguous()
    # pred_norm: b x h*w x c
    pred_norm = pred_norm.view(b*h*w, c)
    # pred_norm: b*h*w x c
    pred_norm = pred_norm.view(b*h*w, c, 1)
    # pred_norm: b*h*w x c x 1
    #
    for j, (cm, label_noisy) in enumerate(zip(cms, labels)):
        # cm: learnt confusion matrix for each noisy label, b x c_r_d x h x w, where c_r_d < c
        # label_noisy: noisy label, b x h x w

        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2

        # reconstruct the full-rank confusion matrix from low-rank approximations:
        cm1 = cm[:, 0:r * c, :, :]
        cm2 = cm[:, r * c:c_r_d-1, :, :]
        scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
        cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
        cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
        cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)

        # add an identity residual to make approximation easier
        identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
        cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
        cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)

        # calculate noisy prediction from confusion matrix and prediction
        pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        regularisation_ = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)

        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())

        regularisation += regularisation_

        main_loss += loss_current

    regularisation = alpha*regularisation

    loss = main_loss + regularisation

    return loss, main_loss, regularisation