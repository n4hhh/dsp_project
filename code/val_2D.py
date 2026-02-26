import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

from utils import metrics as metrics_utils


def calculate_metric_percase(pred, gt):
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], surface_dice_tolerance=1, return_ece=False):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    probability = np.zeros((classes,) + label.shape, dtype=np.float32)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            logits = net(input)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            out = np.argmax(probs, axis=0)
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
            for c in range(classes):
                prob_c = zoom(probs[c], (x / patch_size[0], y / patch_size[1]), order=1)
                probability[c, ind] = prob_c
    metric_list = []
    for i in range(1, classes):
        dice, hd95 = calculate_metric_percase(prediction == i, label == i)
        surf_dice = metrics_utils.surface_dice(
            prediction == i, label == i, tolerance=surface_dice_tolerance)
        metric_list.append((dice, hd95, surf_dice))
    if return_ece:
        ece = metrics_utils.expected_calibration_error(probability, label)
        return metric_list, ece
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256], surface_dice_tolerance=1, return_ece=False):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    probability = np.zeros((classes,) + label.shape, dtype=np.float32)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            probs = torch.softmax(output_main, dim=1).squeeze(0).cpu().numpy()
            out = np.argmax(probs, axis=0)
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
            for c in range(classes):
                prob_c = zoom(probs[c], (x / patch_size[0], y / patch_size[1]), order=1)
                probability[c, ind] = prob_c
    metric_list = []
    for i in range(1, classes):
        dice, hd95 = calculate_metric_percase(prediction == i, label == i)
        surf_dice = metrics_utils.surface_dice(
            prediction == i, label == i, tolerance=surface_dice_tolerance)
        metric_list.append((dice, hd95, surf_dice))
    if return_ece:
        ece = metrics_utils.expected_calibration_error(probability, label)
        return metric_list, ece
    return metric_list
