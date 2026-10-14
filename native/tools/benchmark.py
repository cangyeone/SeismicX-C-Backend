# SPDX-License-Identifier: GPL-3.0-only
"""End-to-end warm median/p95 latency including preprocessing and synchronized GPU IO."""
import argparse,json,os,platform,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'tools'))
from pnsn import Picker,load_library
from validate import normalize,picks

def timed(fn,x,repeats):
    begin=time.perf_counter();fn(x);cold=time.perf_counter()-begin
    for _ in range(3):fn(x)
    times=[]
    for _ in range(repeats):
        begin=time.perf_counter();out=fn(x);times.append((time.perf_counter()-begin)*1000)
    return out,dict(cold_ms=cold*1000,median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),repeats=repeats)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mlx',action='store_true');p.add_argument('--cuda',action='store_true');p.add_argument('--repeats',type=int,default=30);p.add_argument('--input',help='[N,3] npy, raw input');p.add_argument('--threads',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args()
    if a.repeats<1:raise ValueError('repeats must be positive')
    import torch
    from export import load_network
    torch.set_num_threads(a.threads);model=load_network(ROOT.parent/'ckpt/pnsn.v3.pt');rng=np.random.default_rng(20261014)
    source=np.load(a.input) if a.input else None
    def torch_forward(x):
        with torch.no_grad():return model(torch.from_numpy(normalize(x).T.copy())[None])[0].T.numpy()
    devices=['cpu']+(['cuda'] if a.cuda else [])
    records=[];lib=load_library()
    for length in (6144,10240):
        x=rng.normal(size=(length,3)) if source is None else np.pad(source[:length],((0,max(0,length-len(source))),(0,0)),mode='edge')
        model=model.cpu();reference,timing=timed(torch_forward,x,a.repeats)
        reference_picks=picks(reference,lib)
        records.append(dict(backend='torch-cpu',precision='fp32',length=length,**timing))
        if a.cuda:
            model=model.cuda(0)
            def torch_gpu(x):
                with torch.no_grad():
                    y=model(torch.from_numpy(normalize(x).T.copy())[None].cuda(0))[0].T.cpu().numpy()
                    torch.cuda.synchronize(0);return y
            y,timing=timed(torch_gpu,x,a.repeats);records.append(dict(backend='torch-cuda',precision='fp32',length=length,**timing))
        for backend in ['c-cpu']+(['c-cuda'] if a.cuda else [])+(['mlx'] if a.mlx else []):
            for mode in ('fp32','int8'):
                library=None
                if backend=='c-cuda':library=str(ROOT/'build/libpnsn_cuda.so')
                if backend=='mlx':
                    from pnsn_mlx import MLXPicker
                    picker=MLXPicker(ROOT/f'models/pnsn-{mode}.bin',window=length,overlap=1024)
                else:picker=Picker(ROOT/f'models/pnsn-{mode}.bin',library,window=length,overlap=1024,threads=a.threads)
                y,timing=timed(picker.predict,x,a.repeats)
                assert np.isfinite(y).all(), (backend,mode,"nonfinite output")
                error=abs(y-reference)
                got=picks(y,lib);remaining=list(got);matched=[]
                for phase,at,score in reference_picks:
                    nearby=[q for q in remaining if q[0]==phase and abs(q[1]-at)<=50]
                    if nearby:
                        best=min(nearby,key=lambda q:abs(q[1]-at));remaining.remove(best);matched.append(abs(best[1]-at))
                records.append(dict(backend=backend,precision=mode,length=length,**timing,
                    probability_max_error=float(error.max()),probability_mean_error=float(error.mean()),
                    reference_picks=len(reference_picks),picks=len(got),matches_within_50_samples=len(matched),
                    max_matched_sample_error=max(matched,default=None)))
                if backend!='mlx':picker.close()
    result=dict(host=platform.node(),platform=platform.platform(),cpu_features=lib.pnsn_cpu_features().decode(),threads=a.threads,
                waveform='seeded synthetic' if source is None else str(a.input),records=records)
    if a.cuda:result['gpu']=torch.cuda.get_device_name(0)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
