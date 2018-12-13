# -*- coding: utf-8 -*-

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

# Sensitivity = TP/(TP+FN)
def sensitivity(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(np.logical_and(y_true,y_pred).astype(float))
    fn = np.sum(np.logical_and(y_true,np.logical_not(y_pred)).astype(float))
    return tp/(tp+fn)

# Specificity = TN/(TN+FP)
def specificity(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tn = np.sum(np.logical_and(np.logical_not(y_true),np.logical_not(y_pred)).astype(float))
    fp = np.sum(np.logical_and(np.logical_not(y_true),y_pred).astype(float))
    return tn/(tn+fp)

# Accuracy = (TP+TN)/(TP+TN+FP+FN)
def accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(np.logical_and(y_true,y_pred).astype(float))
    fn = np.sum(np.logical_and(y_true,np.logical_not(y_pred)).astype(float))
    tn = np.sum(np.logical_and(np.logical_not(y_true),np.logical_not(y_pred)).astype(float))
    fp = np.sum(np.logical_and(np.logical_not(y_true),y_pred).astype(float))
    return (tp+tn)/(tp+tn+fp+fn)

def dice_coeff(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred).astype(float)
    score = (2. * intersection) / (
      np.sum(y_true) + np.sum(y_pred))
    return score

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    
    img = np.ascontiguousarray(img)
    
    U = unary_from_softmax(output_probs)

    d.setUnaryEnergy(U)

#     d.addPairwiseGaussian(sxy=10, compat=1)
#     d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img, compat=10)

    d.addPairwiseGaussian(sxy=10, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=10, rgbim=img, compat=3)
    
    
    Q = d.inference(5)
    res = np.argmax(np.array(Q), axis=0).reshape((h, w))
#     res = np.array(Q).reshape((h,w))
    
    return res