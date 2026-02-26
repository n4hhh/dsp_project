#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric
from scipy.ndimage import binary_erosion, distance_transform_edt


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def surface_dice(pred, gt, tolerance=1):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)
    pred_surface_sum = pred_surface.sum()
    gt_surface_sum = gt_surface.sum()
    if pred_surface_sum == 0 and gt_surface_sum == 0:
        return 1.0
    if pred_surface_sum == 0 or gt_surface_sum == 0:
        return 0.0
    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt = distance_transform_edt(~gt_surface)
    pred_to_gt = (dt_gt[pred_surface] <= tolerance).sum()
    gt_to_pred = (dt_pred[gt_surface] <= tolerance).sum()
    return (pred_to_gt + gt_to_pred) / (pred_surface_sum + gt_surface_sum)


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = np.max(probs, axis=0)
    pred = np.argmax(probs, axis=0)
    correct = (pred == labels).astype(np.float32)
    conf = conf.flatten()
    correct = correct.flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            ece += (mask.mean()) * np.abs(acc - avg_conf)
    return float(ece)