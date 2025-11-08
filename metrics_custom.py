# Librerias
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# métrica de accuracy con threshold que acepta tensores
def accuracy_trch (tensor_pred, tensor_true, threshold = 0.5):
    yhat = (tensor_pred.detach().numpy()>=threshold).astype(int)
    y = tensor_true.detach().numpy().astype(int)
    accuracy = accuracy_score(y, yhat)
    return accuracy

# métrica de matriz de confusión con threshold que acepta tensores
def confusion_trch (tensor_pred, tensor_true, threshold = 0.5):
    y = tensor_true.detach().numpy()
    yhat = (tensor_pred.detach().numpy()>=threshold).astype(float)
    cm = confusion_matrix(y, yhat)
    return cm

def metrics_semseg(tensor_pred, tensor_true):
    yhat = tensor_pred.detach().numpy().astype(int)
    y = tensor_true.detach().numpy().astype(int)
    accuracy = accuracy_score(y.flatten(), yhat.flatten())

    yhat_1 = np.where(yhat == 1, 1, 0)
    yhat_2 = np.where(yhat == 2, 1, 0)
    y_1 = np.where(y == 1, 1, 0)
    y_2 = np.where(y == 2, 1, 0)

    union_1 = np.where(yhat_1 + y_1 >= 1, 1, 0)
    union_2 = np.where(yhat_2 + y_2 >= 1, 1, 0)
    inter_1 = np.where(yhat_1 + y_1 == 2, 1, 0)
    inter_2 = np.where(yhat_2 + y_2 == 2, 1, 0)

    iou_1 = np.sum(inter_1)/np.sum(union_1)
    iou_2 = np.sum(inter_2)/np.sum(union_2)
    
    return accuracy, iou_1, iou_2