# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
"""Evaluate a CRUNCH model on labeled raw trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
import joblib
from predict import features,HARMONIC,ROOT
from original_core import predict
from metrics import score

def evaluate(model_path,fields,observations,output):
    artifact=joblib.load(model_path)
    if artifact['model']!='CRUNCH' or artifact['harmonic_columns']!=HARMONIC or artifact['threshold']!=.5:
        raise ValueError('Expected the fixed-threshold CRUNCH architecture')
    data=features(fields,observations)
    if 'label' not in data or not data.label.isin(['grain','silage']).all():raise ValueError('Evaluation requires grain/silage labels')
    prob,branch=predict(data,artifact['branches'])
    result=data[['field_year_id','label']].copy();result['silage_score']=prob;result['branch']=branch
    result['prediction']=np.where(np.isnan(prob),'uncertain',np.where(prob>=.5,'silage','grain'))
    supported=np.isfinite(prob)
    metrics=dict(model='CRUNCH',threshold=.5,total_rows=len(result),scored_rows=int(supported.sum()),
                 coverage=float(supported.mean()),**(score(result[supported]) if supported.any() else {}))
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'predictions.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2));return result,metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Evaluate CRUNCH at fixed threshold 0.5')
    p.add_argument('--model',type=Path,default=ROOT/'models/crunch.joblib')
    p.add_argument('--fields',type=Path,default=ROOT/'data/test/fields.csv')
    p.add_argument('--observations',type=Path,default=ROOT/'data/test/observations.csv')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation')
    a=p.parse_args();evaluate(a.model,a.fields,a.observations,a.output)
