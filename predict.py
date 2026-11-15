# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from original_core import EVENT,RELATIVE,INDICES,optical_features,predict
ROOT=Path(__file__).resolve().parent
HARMONIC=[f'harmonic_{c}_{term}' for c in ['ndvi','lswi','ndti'] for term in ['intercept','cos1','sin1','cos2','sin2']]
def features(field_path,observation_path):
    fields=pd.read_csv(field_path);observations=pd.read_csv(observation_path)
    if not {'field_year_id','year'}<=set(fields):raise ValueError('fields requires field_year_id,year')
    if not {'field_year_id','date',*INDICES}<=set(observations):raise ValueError('observations requires field_year_id,date,ndvi,lswi,ndti')
    if fields.field_year_id.isna().any() or fields.field_year_id.duplicated().any():raise ValueError('Field IDs must be unique and nonmissing')
    if fields.year.isna().any() or (fields.year%1!=0).any():raise ValueError('year must be an integer')
    observations['date']=pd.to_datetime(observations.date,errors='raise')
    if observations.date.isna().any():raise ValueError('Missing observation date')
    if observations.duplicated(['field_year_id','date']).any():raise ValueError('Duplicate field/date observation')
    if not observations.field_year_id.isin(fields.field_year_id).all():raise ValueError('Observation ID missing from fields')
    years=observations.field_year_id.map(fields.set_index('field_year_id').year)
    if not observations.date.dt.year.eq(years).all():raise ValueError('Observation year differs from field year')
    observations[list(INDICES)]=observations[list(INDICES)].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    groups=dict(tuple(observations.groupby('field_year_id')));records=[]
    for row in fields.itertuples():
        g=groups.get(row.field_year_id,observations.iloc[:0])
        optics=optical_features(g)
        valid=g[g.date.dt.month.between(5,11)].dropna(subset=list(INDICES)).sort_values('date')
        harmonic=np.full(15,np.nan)
        if len(valid):
            days=(valid.date-pd.Timestamp(f'{int(row.year)}-01-01')).dt.days.to_numpy()
            phase=2*np.pi*days/365.25
            a=np.column_stack([np.ones(len(days)),np.cos(phase),np.sin(phase),np.cos(2*phase),np.sin(2*phase)])
            penalty=np.diag([0.,1.,1.,1.,1.])
            coef=np.linalg.solve(a.T@a/len(a)+.01*penalty,a.T@valid[['ndvi','lswi','ndti']].to_numpy()/len(a))
            harmonic=coef.T.ravel()
        records.append(dict(field_year_id=row.field_year_id,**optics,**dict(zip(HARMONIC,harmonic))))
    # Never trust precomputed predictor columns in the metadata input.
    meta=fields.drop(columns=[c for c in records[0] if c!='field_year_id' and c in fields]) if records else fields
    return meta.merge(pd.DataFrame(records),on='field_year_id',validate='one_to_one')

def infer(args):
    data=features(args.fields,args.observations)
    artifact=joblib.load(args.model)
    if artifact['model']!='CRUNCH' or artifact['harmonic_columns']!=HARMONIC:raise ValueError('Incompatible model artifact')
    prob,branch=predict(data,artifact['branches'])
    out=data[['field_year_id','year','optical_supported','valid_dates']].copy()
    out['silage_score']=prob;out['branch']=branch
    out['prediction']=np.where(np.isnan(prob),'uncertain',np.where(prob>=.5,'silage','grain'))
    args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    print(json.dumps(dict(output=str(args.output),rows=len(out),scored=int(np.isfinite(prob).sum()))))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='CRUNCH: corn end-use recognition (WA training, MD test)')
    parser.add_argument('--fields',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=ROOT/'models/crunch.joblib')
    infer(parser.parse_args())
