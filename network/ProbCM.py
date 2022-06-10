import torch
import torch.nn as nn
import math
from torch import distributions
import matplotlib.pyplot as plt
from .unet_cm import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Conv1x1Decoder(nn.Module):

    def __init__(self,
                 latent_dim,
                 init_features,
                 num_classes=1,
                 num_1x1_convs=3):
        super().__init__()

        self._num_1x1_convs = num_1x1_convs
        self._latent_dim = latent_dim
        self._num_classes = num_classes
        self._features = init_features

        self.net = self._build()


    def forward(self, z, unet_features):
        """ Add the noise to the output of the UNet model, then pass it through several 1x1 convolutions
            z: [Batch size, latent_dim]
            unet_feature: [Batch size, input_channels, H, W]
        """

        *_, h, w = unet_features.shape
        out = torch.cat([unet_features, z[..., None, None].tile(dims=(1, 1, h, w))], dim=1)
        logits = self.net(out)

        return logits

    def _build(self):
        layers = []
        in_channels = self._latent_dim + self._features
        for i in range(self._num_1x1_convs - 1):
            layers += [nn.Conv2d(in_channels, in_channels, (1, 1)),
                       nn.LeakyReLU(0.1)]

        layers += [nn.Conv2d(in_channels, self._num_classes, (1, 1))]


        return nn.Sequential(*layers)


class AxisAlignedConvGaussian(nn.Module):
    """
    Takes in RGB image and a segmentation ground truth.
    Outputs the mean and log std of of the input.
    """

    def __init__(self,
                 latent_dim,
                 in_channels=3,
                 init_features=32,
                 ):

        super().__init__()
        self._latent_dim = latent_dim
        features = init_features
        self.encoder1 = AxisAlignedConvGaussian._block(in_channels, features)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = AxisAlignedConvGaussian._block(features, 2 * features)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = AxisAlignedConvGaussian._block(features * 2, features * 4)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = AxisAlignedConvGaussian._block(features * 4, features * 8)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = AxisAlignedConvGaussian._block(features * 8, features * 8)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = AxisAlignedConvGaussian._block(features * 8, features * 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self._mu_log_sigma = nn.Conv2d(8 * features, 2 * self._latent_dim, (1, 1))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        bottleneck = self.bottleneck(self.pool5(enc5))

        mu_log_sigma = self._mu_log_sigma(self.avg_pool(bottleneck))
        mu = mu_log_sigma[:, :self._latent_dim, 0, 0]
        log_sigma = mu_log_sigma[:, self._latent_dim:, 0, 0]
# distributions.Independent(distributions.Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return mu, log_sigma


    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.1)
        )

class UNet(nn.Module):

    def __init__(self, in_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = UNet._block(features * 8, features * 8)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 8)

        self.upconv5 = nn.ConvTranspose2d(
            features * 8, features * 8, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder5 = UNet._block(features * 8 * 2, features * 8)

        self.upconv4 = nn.ConvTranspose2d(
            features * 8, features * 8, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder4 = UNet._block(features * 8 * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder3 = UNet._block(features * 4 * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder2 = UNet._block(features * 2 * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=(2, 2), stride=(2, 2)
        )
        self.decoder1 = UNet._block(features * 2, features)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return dec1

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.01)
        )


class ProbCMNet(nn.Module):


    def __init__(self,
                 latent_dim,
                 in_channels,
                 num_classes,
                 low_rank=True,
                 num_1x1_convs=3,
                 init_features=32,
                 lq=0.7):
        super().__init__()
        self._latent_dim = latent_dim
        self._unet = UNet(in_channels, init_features)
        self._f_comb = Conv1x1Decoder(latent_dim, init_features, num_classes, num_1x1_convs)
        # self._prior = AxisAlignedConvGaussian(latent_dim, in_channels, init_features)  # RGB image
        self._posterior = AxisAlignedConvGaussian(latent_dim, in_channels + 1, init_features)  # Image + ground truth
        self.low_rank = low_rank
        self.validation = False
        self.lq = lq

        if self.low_rank is False:
            self.decoders_noisy_layers = cm_layers(in_channels=latent_dim, norm='in', class_no=num_classes)
            #self.decoders_noisy_layers = gcm_layers(num_classes,256,256)
        else:
            self.decoders_noisy_layers=low_rank_cm_layers(in_channels=latent_dim, norm='in', class_no=num_classes, rank=1)


    def forward(self, *args):

        if self.training:
                
                img, mask = args
                self.mu, self.log_sigma = self._posterior(torch.cat([img, mask], dim=1))
                self.q = distributions.Normal(self.mu, torch.exp(self.log_sigma) + 1e-3)
                z_q = self.q.sample()
                unet_features = self._unet(img)
                logits = self._f_comb(z_q, unet_features)
                h,w = img.shape[-2:]
                y_noisy = self.decoders_noisy_layers(z_q[..., None, None].tile(dims=(1, 1, h, w)).detach())

                return logits,y_noisy

        else:
            img = args[0]
            batch_size = img.shape[0]
            mean = torch.zeros(batch_size, self._latent_dim, device=img.device)
            cov = torch.eye(self._latent_dim, device=img.device)
            prior = distributions.MultivariateNormal(mean, cov)
            z_p = prior.sample()
            unet_features = self._unet(img)
            logits = self._f_comb(z_p, unet_features)
            if self.validation:
                h,w = img.shape[-2:]
                cm = self.decoders_noisy_layers(z_p[..., None, None].tile(dims=(1, 1, h, w)).detach())
                #pred_noisy = self.pred_noise_low_rank(logits, cm)
                pred_noisy = self.pred_noisy(logits, cm)
                return pred_noisy
            else:
                return logits

    @staticmethod
    def init_weight(m):

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(m.weight)

            if hasattr(m, 'bias'):
                torch.nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.GroupNorm):
            torch.nn.init.normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def kl(self):
        kld = torch.mean(-0.5 * torch.sum(1 + 2 * self.log_sigma - self.mu ** 2 - self.log_sigma.exp()**2, dim=1))
        return kld
    
    @staticmethod
    def reconstruction_loss(pred, target):
        return nn.CrossEntropyLoss()(pred, target)
    
    def warmup1(self, pred, target, beta=0.01):

        kl = self.kl()
        recon_loss = self.reconstruction_loss(pred, target)
        
        return beta * kl + recon_loss 
    
    def warmup2(self, cms):
        
        if self.low_rank:
            return -trace_low_rank(cms)
        else:
            return -trace_reg(cms)
    
    def elbo1(self, pred, cms, target, alpha=1, beta=0.01):

        kl = self.kl()
        recon_loss = self.reconstruction_loss(pred, target)
        if self.low_rank:
            _,_,TR = noisy_label_loss_low_rank(pred, [cms], [target], alpha)
        else:
            _,_,TR = noisy_label_loss(pred, [cms], [target], alpha)

        #print("Kl  ", kl.item(), "Recon loss  ", recon_loss.item())

        return beta * kl + recon_loss - TR
    
    def elbo(self, pred, cms, target, alpha=2, beta=0.01):

        kl = self.kl()
        if self.low_rank:
            recon_loss,CE,TR = noisy_label_loss_low_rank(pred, [cms], [target], alpha)
        else:
            recon_loss,CE,TR = noisy_label_loss(pred, [cms], [target], alpha)
        
        #print("Kl  ", kl.item(), "Recon loss  ", recon_loss.item())

        return beta * kl + recon_loss + TR + Lq_loss(pred,target,self.lq)
    
    def pred_noisy(self, pred, cm):

        b, c, h, w = pred.size()

        # normalise the segmentation output tensor along dimension 1
        pred_norm = nn.Softmax(dim=1)(pred)

        # b x c x h x w ---> b*h*w x c x 1
        pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

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

        return pred_noisy
    
    def pred_noise_low_rank(self, pred, cm):

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
        identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device=pred.device, dtype=torch.float32)
        cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
        cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)

        # calculate noisy prediction from confusion matrix and prediction
        pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        return pred_noisy


def trace_reg(cm):
    
    b, c, h, w = cm.size()
    c = int(math.sqrt(c))
    cm = cm.view(b, c**2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

    # normalisation along the rows:
    cm = cm / cm.sum(1, keepdim=True)

    regularisation = torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)

    return regularisation

def trace_low_rank(cm):
    
    b, c_r_d, h, w = cm.size()
    c = int(math.sqrt(c_r_d-1))
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

    regularisation_ = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)
    
    return regularisation_


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


             
def Lq_loss(logits, targets, q=0.7):
    p = F.softmax(logits, dim=1)

    Yg = torch.gather(p, 1, targets.unsqueeze(1))

    loss = (1-(Yg**q))/q
    loss = torch.mean(loss)

    return loss