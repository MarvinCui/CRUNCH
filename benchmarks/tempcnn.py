# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TIANYANG LIU and BORAN CUI
"""TempCNN architecture port; differences from original execution in PROTOCOL.txt."""
import json, random, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent

class TempCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[]
        for channels in [3,64,64]:
            blocks.extend([nn.Conv1d(channels,64,5,padding=2),
                nn.BatchNorm1d(64,eps=1e-3,momentum=.01),nn.ReLU(),nn.Dropout(.5)])
        self.conv=nn.Sequential(*blocks)
        self.hidden=nn.Sequential(nn.Linear(32*64,256),
            nn.BatchNorm1d(256,eps=1e-3,momentum=.01),nn.ReLU(),nn.Dropout(.5))
        self.output=nn.Linear(256,2)
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self,x):
        # Match the original Keras time-major Flatten, with PyTorch channel-first convs.
        x=self.conv(x).transpose(1,2).reshape(len(x),-1)
        return self.output(self.hidden(x))

    def penalty(self):
        return 1e-6*sum(m.weight.square().sum() for m in self.modules()
                        if isinstance(m,(nn.Conv1d,nn.Linear)))

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def standardize(x,mask):
    mean=x[mask].astype('float64').mean(axis=(0,2))
    std=np.maximum(x[mask].astype('float64').std(axis=(0,2)),1e-6)
    return ((x-mean[None,:,None])/std[None,:,None]).astype('float32'),mean,std

def fit(x,y,mask,mode,seed,epochs,valid=None):
    seed_all(seed)
    xt=torch.as_tensor(x[mask],device='cuda');yt=torch.as_tensor(y[mask],device='cuda')
    weights=torch.ones(2,device='cuda')
    if mode=='balanced':weights=len(yt)/(2*torch.bincount(yt).float())
    model=TempCNN().cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8)
    best=float('inf');best_epoch=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(yt),device='cuda');total=0.
        for indices in order.split(32):
            assert len(indices)>1, 'BN requires more than one example in final batch'
            opt.zero_grad(set_to_none=True)
            loss=(F.cross_entropy(model(xt[indices]),yt[indices],reduction='none')*weights[yt[indices]]).mean()+model.penalty()
            loss.backward();opt.step();total+=float(loss.detach())*len(indices)
        row=dict(epoch=epoch,training_loss=total/len(yt))
        if valid is not None:
            model.eval()
            with torch.no_grad():
                val_loss=float(F.cross_entropy(model(torch.as_tensor(x[valid],device='cuda')),
                    torch.as_tensor(y[valid],device='cuda'))+model.penalty())
            row['validation_loss']=val_loss
            history.append(row)
            if val_loss < best:best=val_loss;best_epoch=epoch
            else:break  # Original patience=0; checkpoint's best epoch controls fresh full-source refit.
        else:history.append(row);best_epoch=epoch
    torch.cuda.synchronize()
    return model,best_epoch,history

def main():
    if not torch.cuda.is_available():raise RuntimeError('Run this experiment through the Slurm GPU allocation.')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    z=np.load(ROOT/'inputs.npz');fields=pd.read_csv(ROOT/'fields.csv');x=z['x'];y=z['y']
    train=z['train'];inner=z['inner_train'];valid=z['inner_valid']
    assert np.all((inner|valid)==train) and not np.any(inner&valid)
    xs,_,_=standardize(x,inner);xf,mean,std=standardize(x,train)
    hardware=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  cudnn_allow_tf32=False,matmul_allow_tf32=False,deterministic_algorithms=True)
    (ROOT/'results/tempcnn_environment.json').write_text(json.dumps(hardware,indent=2))
    predictions=[];runs=[];histories=[]
    for mode in ['default','balanced']:
        for seed in [17,37,66]:
            start=time.perf_counter()
            probe,epochs,hist=fit(xs,y,inner,mode,seed,20,valid)
            selection_seconds=time.perf_counter()-start
            del probe
            start=time.perf_counter();model,_,refit_hist=fit(xf,y,train,mode,seed,epochs)
            fit_seconds=time.perf_counter()-start
            model.eval();torch.cuda.synchronize();start=time.perf_counter()
            for target in ['md']:
                mask=z[target]
                with torch.no_grad():scores=model(torch.as_tensor(xf[mask],device='cuda')).softmax(1)[:,1].cpu().numpy()
                p=fields.loc[mask,['field_year_id','physical_field','label']].copy()
                p['model']='TempCNN';p['mode']=mode;p['seed']=seed;p['target']=target;p['silage_score']=scores
                predictions.append(p)
            torch.cuda.synchronize();predict_seconds=time.perf_counter()-start
            file=ROOT/f'results/TempCNN_{mode}_{seed}.pt'
            state={k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,mean=mean.tolist(),std=std.tolist(),
                train_ids=z['ids'][train].tolist(),inner_training_ids=z['ids'][inner].tolist(),
                inner_validation_ids=z['ids'][valid].tolist(),epochs=epochs,mode=mode,seed=seed),file)
            # Confirm that serialization preserves scores on this GPU.
            check=TempCNN().cuda();check.load_state_dict(torch.load(file,weights_only=True)['state_dict']);check.eval()
            with torch.no_grad():
                actual=check(torch.as_tensor(xf[z['md']],device='cuda')).softmax(1)[:,1].cpu().numpy()
            assert np.array_equal(actual,scores)
            runs.append(dict(model='TempCNN',mode=mode,seed=seed,selected_epochs=epochs,
                epoch_selection_seconds=selection_seconds,fit_seconds=fit_seconds,
                predict_MD_seconds=predict_seconds,parameter_count=sum(p.numel() for p in model.parameters()),
                dimension=256,serialized_bytes=file.stat().st_size))
            for phase,rows in [('internal',hist),('refit',refit_hist)]:
                histories.extend(dict(mode=mode,seed=seed,phase=phase,**r) for r in rows)
            pd.concat(predictions).to_csv(ROOT/'results/tempcnn_predictions.csv',index=False)
            pd.DataFrame(runs).to_csv(ROOT/'results/tempcnn_runs.csv',index=False)
            pd.DataFrame(histories).to_csv(ROOT/'results/tempcnn_epochs.csv',index=False)
            print(json.dumps(runs[-1]),flush=True)

if __name__=='__main__':main()
