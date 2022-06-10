# -*- coding:utf-8 -*-
import time
import random, glob
import numpy as np
import numpy.linalg as npl
import cv2

from batchgenerators.transforms import GammaTransform, Compose
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform


def get_DA(patch_size,intensity_DA=False,spatial_DA=False):
    
    additive_brightness_mu = 0
    additive_brightness_sigma = 0.2
    additive_brightness_p_per_sample = 0.3
    gamma_range = (0.5, 1.6)


    tr_transforms = []

    if intensity_DA:
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=False, p_per_sample=0.2))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.65, 1.5), p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=False,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25))
        tr_transforms.append(GammaTransform(gamma_range, True, True, retain_stats=True,p_per_sample=0.15))  # inverted gamma

        tr_transforms.append(BrightnessTransform(additive_brightness_mu,additive_brightness_sigma,
                                                 False, p_per_sample=additive_brightness_p_per_sample))
    if spatial_DA:
        tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=False, random_crop=False,
            p_el_per_sample=0.1, p_rot_per_sample=0.3,
            )
        )
        tr_transforms.append(MirrorTransform(axes=(0, 1)))

    tr_transforms = Compose(tr_transforms)
    
    return tr_transforms

