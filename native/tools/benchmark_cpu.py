"""Actual single-window CPU latency, raw [3,L] input with internal preprocessing."""
import argparse,json,platform,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pnsn import Picker
p=argparse.ArgumentParser();p.add_argument('--model',choices=['pnsn','china-unet'],default='pnsn');p.add_argument('--repeats',type=int,default=50);p.add_argument('--output',required=True);a=p.parse_args()
if a.repeats<1:raise ValueError('repeats must be positive')
rows=[];rng=np.random.default_rng(20261014)
for length in [6144,10240]:
 x=rng.standard_normal((1,3,length),dtype=np.float32)
 for mode in ['fp32','int8']:
  with Picker(ROOT/f'models/{a.model}-{mode}.bin',window=length,overlap=1024,threads=1) as picker:
   for _ in range(3):picker.predict_batch(x)
   times=[]
   for _ in range(a.repeats):
    start=time.perf_counter();out=picker.predict_batch(x);times.append((time.perf_counter()-start)*1000)
   assert np.isfinite(out).all() and abs(out.sum(1)-1).max()<2e-6
   median=float(np.median(times));rows.append(dict(model=mode,length=length,batch=1,threads=1,median_ms=median,p95_ms=float(np.percentile(times,95)),windows_per_second=1000/median,cpu_features=picker.lib.pnsn_cpu_features().decode(),times_ms=times))
result=dict(host=platform.node(),model=a.model,platform=platform.platform(),scope='single CPU raw host input to probabilities; internal float64 preprocessing and output allocation included; no NMS; 3 warmups then measured repeats',records=rows)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**result,'records':[{k:v for k,v in r.items() if k!='times_ms'} for r in rows]},indent=2))
