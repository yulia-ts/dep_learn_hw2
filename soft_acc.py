import numpy as np
import torch
import torch.nn as nn


def count_soft_acc(pred_exp, label_vec):
    acc = 0.000
    for i,v in enumerate(pred_exp):
        acc += label_vec[i][v]
    return acc