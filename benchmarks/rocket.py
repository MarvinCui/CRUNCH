# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
import sys,time,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from aeon.classification.convolution_based import RocketClassifier,MiniRocketClassifier,MultiRocketClassifier
ROOT=Path(__file__).resolve().parent
z=np.load(ROOT/'inputs.npz');fields=pd.read_csv(ROOT/'fields.csv');train=z['train'];x=z['x'].astype('float64');y=z['y']
mean=x[train].mean(axis=(0,2));std=np.maximum(x[train].std(axis=(0,2)),1e-6);x=((x-mean[None,:,None])/std[None,:,None]).astype('float32')
parts=[];history=[]
for name,klass in [('ROCKET',RocketClassifier),('MiniRocket',MiniRocketClassifier),('MultiRocket',MultiRocketClassifier)]:
 for mode in ['default','balanced']:
  for seed in [17,37,66]:
   t=time.perf_counter();model=klass(n_kernels=10000,n_jobs=1,random_state=seed,class_weight='balanced' if mode=='balanced' else None)
   model.fit(x[train],y[train]);fit_seconds=time.perf_counter()-t
   t=time.perf_counter()
   for target in ['md']:
    mask=z[target];decision=model.pipeline_.decision_function(x[mask]);prob=expit(decision)
    assert np.array_equal(decision>0,model.predict(x[mask]).astype(bool))
    p=fields.loc[mask,['field_year_id','physical_field','label']].copy();p['model']=name;p['mode']=mode;p['seed']=seed;p['target']=target;p['silage_score']=prob;parts.append(p)
   elapsed=time.perf_counter()-t
   file=ROOT/f'results/{name}_{mode}_{seed}.joblib';joblib.dump(dict(classifier=model,mean=mean,std=std,train_ids=z['ids'][train]),file)
   history.append(dict(model=name,mode=mode,seed=seed,fit_seconds=fit_seconds,predict_MD_seconds=elapsed,
       dimension=int(model.estimator_.coef_.shape[-1]),selected_ridge_alpha=float(model.estimator_.alpha_),serialized_bytes=file.stat().st_size))
   pd.concat(parts).to_csv(ROOT/'results/rocket_predictions.csv',index=False);pd.DataFrame(history).to_csv(ROOT/'results/rocket_runs.csv',index=False)
   print(json.dumps(history[-1]),flush=True)
