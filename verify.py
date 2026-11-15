# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
"""Check release integrity, split isolation, saved inference and full raw-data refit."""
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from predict import features,HARMONIC,ROOT
from original_core import predict
from train import train
from metrics import score

def verify(refit=False):
    count=0
    for line in (ROOT/'CHECKSUMS.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Checksum mismatch: '+name)
        count+=1
    tr=pd.read_csv(ROOT/'data/train/fields.csv');te=pd.read_csv(ROOT/'data/test/fields.csv')
    assert len(tr)==599 and len(te)==44
    assert tr.region.eq('US-WA').all() and te.region.eq('US-MD').all()
    assert tr.label.value_counts().to_dict()=={'silage':562,'grain':37}
    assert te.label.value_counts().to_dict()=={'grain':32,'silage':12}
    assert not set(tr.field_year_id)&set(te.field_year_id)
    assert not set(tr.physical_field)&set(te.physical_field)
    assert te.physical_field.nunique()==34
    model=joblib.load(ROOT/'models/crunch.joblib')
    assert model['model']=='CRUNCH' and model['threshold']==.5
    assert model['authors']==['TIANYANG LIU','BORAN CUI']
    assert model['training_ids']==tr.field_year_id.tolist()
    assert model['harmonic_columns']==HARMONIC and model['classifier_C']==.3 and model['harmonic_alpha']==.01
    assert [len(v['features']) for v in model['branches'].values()]==[24,33]
    assert [v['training_examples'] for v in model['branches'].values()]==[353,599]
    data=features(ROOT/'data/test/fields.csv',ROOT/'data/test/observations.csv')
    expected=pd.read_csv(ROOT/'reference/test_predictions.csv').set_index('field_year_id').loc[data.field_year_id]
    prob,branch=predict(data,model['branches'])
    error=float(np.max(abs(prob-expected.silage_score.to_numpy())))
    assert error<1e-8 and np.array_equal(branch,expected.branch)
    metrics=score(pd.DataFrame(dict(label=data.label.to_numpy(),silage_score=prob)))
    assert metrics['accuracy']==42/44 and abs(metrics['macro_f1']-.9393939393939394)<1e-12
    report=dict(model='CRUNCH',checksummed_files=count,train_field_years=599,test_field_years=44,
        physical_fields_disjoint=True,maximum_saved_prediction_error=error,**metrics)
    if refit:
        with tempfile.TemporaryDirectory(prefix='crunch-refit-') as temp:
            fresh=train(ROOT/'data/train/fields.csv',ROOT/'data/train/observations.csv',Path(temp)/'crunch.joblib')
            p,b=predict(data,fresh['branches']);delta=float(np.max(abs(p-prob)))
            assert delta<1e-6 and np.array_equal(b,branch) and np.array_equal(p>=.5,prob>=.5)
            report.update(retrained_from_raw_train_data=True,maximum_refit_prediction_error=delta,
                          refit_predictions_and_routes_match=True)
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--retrain',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args();report=verify(a.retrain)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
