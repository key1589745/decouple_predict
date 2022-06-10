# -*- coding:utf-8 -*-

import torch
from torch import nn
import os
import math
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
#import scipy.misc
from dataset import *
import itertools
import copy


def dice_compute(pred, groundtruth):           #batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice=[]
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]


    return np.array(dice,dtype=np.float32)




def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)


def Hausdorff_compute(pred,groundtruth,num_class,spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,num_class, 5))
    surface_distance_results = np.zeros((1,num_class, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(num_class):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)
            

    return overlap_results,surface_distance_results


def multi_dice_iou_compute(pred,label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    # nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, \
    #                    truearg == 4, truearg == 5, truearg == 6, truearg == 7), 1)
    nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, truearg == 4, truearg == 5), 1)
    # truelabel = (truearg == 0) * 550 + (truearg == 1) * 420 + (truearg == 2) * 600 + (truearg == 3) * 500 + \
    #             (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice,Iou


class Evaluator(object):
    def __init__(self,data_loader_vali):
        
        self.vali_loaders = data_loader_vali

    def eval(self, model,client):
        
        num_cls = 2
        total_overlap = np.zeros((1, num_cls, 5))
        res = {}
        #model = copy.deepcopy(model).cpu()
        model = model.cpu()
        model.eval()

        for vali_batch in self.vali_loaders[client]:

            #imgs = torch.from_numpy(vali_batch['data']).cuda(non_blocking=True)
            imgs = torch.from_numpy(vali_batch['data'])
            labs = vali_batch['seg']

            output= model(imgs)

            #if 'Partial' in client:
                #output[:,3] = -1
 
            truemax, truearg0 = torch.max(output, 1, keepdim=False)

            truearg = truearg0.detach().cpu().numpy().astype(np.uint8)
            
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            
            overlap_result, _ = Hausdorff_compute(truearg, labs, num_cls, (1.5,1.5,10,1))

            total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)
            
            #del input, truearg0, truemax
        
        model = model.cuda()
        
        meanDice = np.round(np.mean(total_overlap[1:,1,1], axis=0),4)
        std = np.round(np.std(total_overlap[1:,1,1]),4)
        
        return meanDice, std


class Evaluator_feddan(object):
    def __init__(self,data_loader_vali):
        
        self.vali_loaders = data_loader_vali

    def eval(self, model,client):
        
        num_cls = 2
        total_overlap = np.zeros((1, num_cls, 5))
        res = {}
        #model = copy.deepcopy(model).cpu()
        model = model.cpu()
        model.eval()
        model.validation = True
        for vali_batch in self.vali_loaders[client]:

            #imgs = torch.from_numpy(vali_batch['data']).cuda(non_blocking=True)
            imgs = torch.from_numpy(vali_batch['data'])
            labs = vali_batch['seg']

            output= model(imgs)

            #if 'Partial' in client:
                #output[:,3] = -1
 
            truemax, truearg0 = torch.max(output, 1, keepdim=False)

            truearg = truearg0.detach().cpu().numpy().astype(np.uint8)
            
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            
            overlap_result, _ = Hausdorff_compute(truearg, labs, num_cls, (1.5,1.5,10,1))

            total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)
            
            #del input, truearg0, truemax
        
        model.validation = False
        model = model.cuda()
        
        meanDice = np.round(np.mean(total_overlap[1:,1,1], axis=0),4)
        std = np.round(np.std(total_overlap[1:,1,1]),4)
        
        return meanDice, std
    

