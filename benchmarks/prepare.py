"""Prepare paper-method sequences from CRUNCH's released train/test observations."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
parts=[];arrays=[]
for split in ['train','test']:
    fields=pd.read_csv(PROJECT/f'data/{split}/fields.csv');parts.append(fields)
    observations=pd.read_csv(PROJECT/f'data/{split}/observations.csv',parse_dates=['date'])
    groups=dict(tuple(observations.groupby('field_year_id')))
    for row in fields.itertuples():
        g=groups[row.field_year_id];g=g[g.date.dt.month.between(5,11)].dropna(subset=['ndvi','lswi','ndti']).sort_values('date')
        t=(g.date-pd.Timestamp(f'{row.year}-05-01')).dt.days.to_numpy()
        arrays.append(np.array([np.interp(np.linspace(0,213,32),t,g[c]) for c in ['ndvi','lswi','ndti']]))
fields=pd.concat(parts,ignore_index=True);x=np.array(arrays,dtype='float32')
train=np.arange(len(fields))<len(parts[0]);md=~train
a,b=next(StratifiedGroupKFold(20,shuffle=True,random_state=66).split(x[train],fields.loc[train,'label'],fields.loc[train,'physical_field']))
inner_train=np.zeros(len(fields),bool);inner_valid=inner_train.copy();idx=np.flatnonzero(train)
inner_train[idx[a]]=True;inner_valid[idx[b]]=True
assert not set(fields.loc[inner_train,'physical_field'])&set(fields.loc[inner_valid,'physical_field'])
(ROOT/'results').mkdir(exist_ok=True)
np.savez_compressed(ROOT/'inputs.npz',x=x,y=fields.label.eq('silage').to_numpy(int),train=train,md=md,
    inner_train=inner_train,inner_valid=inner_valid,ids=fields.field_year_id.to_numpy(dtype=str))
fields.to_csv(ROOT/'fields.csv',index=False)
print('input',x.shape,'internal',inner_train.sum(),inner_valid.sum())
