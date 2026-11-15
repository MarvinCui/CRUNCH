# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,roc_auc_score,confusion_matrix
def score(rows):
    y=rows.label.eq('silage').to_numpy(int)
    prob=rows.silage_score.to_numpy();pred=prob>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    both=np.unique(y).size==2
    return dict(n=len(y),grain_n=int(tn+fp),silage_n=int(fn+tp),
                macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)) if both else None,
                balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
                silage_f1=float(f1_score(y,pred,zero_division=0)),accuracy=float(accuracy_score(y,pred)),
                auc=float(roc_auc_score(y,prob)) if both else None,
                grain_to_silage=int(fp),silage_to_grain=int(fn))
