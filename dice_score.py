import torch
from torch import Tensor
import numpy as np
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        TP = torch.dot(input.reshape(-1), target.reshape(-1))
        FN = torch.sum(target) - TP
        FP = torch.sum(input) - TP
        TP_percent = TP*100.0/torch.sum(input)
        FP_2_TP = FP*100.0/TP
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * TP
        dice = (2 * TP + epsilon) / (sets_sum + epsilon)
        return dice,TP_percent,FP_2_TP
    else:
        dice = 0
        TP_percent = 0
        FP_2_TP = 0
        for i in range(input.shape[0]):
            dice_each,TP_percent_each,FP_2_TP_each = dice_coeff(input[i, ...], target[i, ...])
            dice += dice_each
            TP_percent += TP_percent_each
            FP_2_TP += FP_2_TP_each
        return dice / input.shape[0], TP_percent / input.shape[0], FP_2_TP / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice_score_list = []
    TP_percent_list = []
    FP_2_TP_list = []
    for channel in range(input.shape[1]):
        dice,TP_percent,FP_2_TP = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        dice_score_list.append(dice)
        TP_percent_list.append(TP_percent)
        FP_2_TP_list.append(FP_2_TP)
    return torch.tensor(dice_score_list),torch.tensor(TP_percent_list),torch.tensor(FP_2_TP_list)

def multiclass_iou(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    dice_score_list,TP_percent_list,FP_2_TP_list = multiclass_dice_coeff(input, target, reduce_batch_first, epsilon)
    iou_score_list = []
    for i in dice_score_list:
        iou_score_list.append(i/(2-i))
    return torch.tensor(iou_score_list),TP_percent_list,FP_2_TP_list

def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    dice_score_list = multiclass_dice_coeff(input, target, reduce_batch_first=True)
    dice_score_mean = dice_score_list.sum()/len(dice_score_list)
    return 1 - dice_score_mean
