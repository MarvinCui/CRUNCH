"""Summarize regenerated or frozen predictions at the fixed threshold 0.5."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--reference',action='store_true');args=p.parse_args()
folder=ROOT/('reference' if args.reference else 'results')
frames=[pd.read_csv(folder/f'{name}_predictions.csv') for name in ['rocket','tempcnn']]
pred=pd.concat(frames);pred=pred[pred.target.eq('md')];rows=[]
for (model,mode,seed),q in pred.groupby(['model','mode','seed']):
    assert len(q)==44 and q.field_year_id.nunique()==44
    y=q.label.eq('silage');yp=q.silage_score>=.5
    rows.append(dict(model=model,mode=mode,seed=seed,accuracy=accuracy_score(y,yp),macro_f1=f1_score(y,yp,average='macro')))
m=pd.DataFrame(rows);assert len(m)==24
s=m.groupby(['model','mode'])[['accuracy','macro_f1']].agg(['mean','std'])
if args.reference:
    reference=pd.read_csv(folder/'metrics.csv').set_index(['model','mode','seed'])
    assert np.allclose(m.set_index(['model','mode','seed'])[['accuracy','macro_f1']],reference.loc[m.set_index(['model','mode','seed']).index,['accuracy','macro_f1']],atol=1e-12)
else:
    (ROOT/'results').mkdir(exist_ok=True);m.to_csv(ROOT/'results/metrics.csv',index=False);s.to_csv(ROOT/'results/summary.csv')
print(s.to_string())
