import numpy as np
from sklearn.metrics import confusion_matrix, recall_score

def sensitivity_score(y_true, y_pred):
    """
    Чувствительность — recall больных
    """
    return recall_score(y_true, y_pred, average='binary')

def specificity_score(y_true, y_pred):
    """
    Специфичность — recall здоровых
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return -1
    return tn/(tn+fp)

def my_fbeta_score(y_true, y_pred, beta=1):
    sens = sensitivity_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    
    return (1+beta**2)*(spec*sens)/(beta**2*spec + sens)

def return_med_metrics(y_true, y_pred, y_prob):
    sens = sensitivity_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    my_fscore = my_fbeta_score(y_true, y_pred, beta=1.2)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError: 
        roc_auc = -1
    
    return [sens, spec, my_fscore, roc_auc]