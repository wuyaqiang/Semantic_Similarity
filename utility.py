#coding=utf-8
from __future__ import print_function

import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        new_h = torch.Tensor(h.data)
        new_h.zero_()
        del h
        return new_h
    else:
        return tuple(repackage_hidden(v) for v in h)


def computeMeasure(predict, label):
    '''Compute precision, recall, f1 and accuracy'''
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(predict)):
        if predict[i] >= 0.2 and int(label[i]) == 1:
            tp += 1
        elif predict[i] >= 0.2 and int(label[i]) == 0:
            fp += 1
        elif predict[i] < 0.2 and int(label[i]) == 1:
            fn += 1
        else:
            tn += 1

    pre = tp / float(fp + tp + 1e-8)
    rec = tp / float(fn + tp + 1e-8)
    f1 = 2 * pre * rec / (pre + rec + 1e-8)
    acc = (tn + tp) / float(tn + fp + fn + tp + 1e-8)

    return pre, rec, f1, acc