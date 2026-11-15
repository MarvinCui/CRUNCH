# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
"""Train CRUNCH on labeled trajectories, using only the provided training files."""
import argparse,json
from pathlib import Path
import joblib
from predict import features,HARMONIC,ROOT
from original_core import fit_model

def train(fields,observations,output):
    if output.resolve()==(ROOT/'models/crunch.joblib').resolve():
        raise ValueError('Preserve the released model; choose an outputs/ path for retraining.')
    data=features(fields,observations)
    if not {'label','region'}<=set(data):raise ValueError('Training metadata requires label and region')
    if not data.label.isin(['grain','silage']).all():raise ValueError('Labels must be grain or silage')
    if not data.optical_supported.all():raise ValueError('Training rows must meet the documented optical support criteria')
    branches=fit_model(data,HARMONIC)
    if set(branches)!={'event','peak_aligned'}:raise ValueError('Both branches require examples of both labels')
    for branch in branches.values():
        branch['training_ids']=data.loc[data[branch['support']].notna().any(axis=1),'field_year_id'].tolist()
        if branch['estimator'][-1].n_iter_.max()>=1000:raise RuntimeError('Logistic regression did not converge')
    artifact=dict(model='CRUNCH',full_name='Corn End-Use Recognition Using Normalized Indices, Change Points, and Harmonics',
        authors=['TIANYANG LIU','BORAN CUI'],version='1.0.0',license='MIT',
        branches=branches,harmonic_columns=HARMONIC,harmonic_order=2,harmonic_alpha=.01,
        classifier_C=.3,threshold=.5,training_ids=data.field_year_id.tolist(),training_count=len(data),
        training_regions=sorted(data.region.unique().tolist()))
    output.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,output)
    print(json.dumps(dict(model='CRUNCH',output=str(output),training_count=len(data),
        branch_counts={k:v['training_examples'] for k,v in branches.items()})))
    return artifact

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train CRUNCH without reading test data')
    p.add_argument('--fields',type=Path,default=ROOT/'data/train/fields.csv')
    p.add_argument('--observations',type=Path,default=ROOT/'data/train/observations.csv')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/crunch_retrained.joblib')
    a=p.parse_args();train(a.fields,a.observations,a.output)
