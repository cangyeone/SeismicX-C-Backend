# SPDX-License-Identifier: GPL-3.0-only
"""Comparable single-window latency on CPU, cuDNN or MLX (real precision labeled)."""
import argparse,json,platform,sys,time,gc
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pnsn import Picker
p=argparse.ArgumentParser();p.add_argument('--model',choices=['pnsn','china-unet'],required=True);p.add_argument('--backend',choices=['cpu','cudnn','cudnn_int8','mlx'],required=True);p.add_argument('--repeats',type=int,default=50);p.add_argument('--output',required=True);a=p.parse_args()
rows=[];rng=np.random.default_rng(20261014)
for length in [6144,10240]:
 x=rng.standard_normal((1,3,length),dtype=np.float32)
 for mode in ['fp32','int8']:
  if a.backend=='mlx':
   import mlx.core as mx
   from pnsn_mlx import MLXPicker
   q=MLXPicker(ROOT/f'models/{a.model}-{mode}.bin',window=length,overlap=1024)
  else:q=Picker(ROOT/f'models/{a.model}-{mode}.bin',None if a.backend=='cpu' else ROOT/f'build/libpnsn_{a.backend}.so',window=length,overlap=1024,threads=1)
  for _ in range(3):q.predict_batch(x)
  times=[]
  for _ in range(a.repeats):
   start=time.perf_counter();out=q.predict_batch(x);times.append((time.perf_counter()-start)*1000)
  assert np.isfinite(out).all() and abs(out.sum(1)-1).max()<2e-6
  median=float(np.median(times));compute='FP32' if mode=='fp32' else 'W8A32' if a.backend=='mlx' else 'Q8 storage / FP32 compute' if a.backend=='cudnn' else 'conv W8A8 / LSTM FP32' if a.backend=='cudnn_int8' and a.model=='pnsn' else 'W8A8 GEMM / float activations'
  rows.append(dict(model=a.model,backend=a.backend,precision=mode,compute=compute,length=length,batch=1,threads=1 if a.backend=='cpu' else None,median_ms=median,p95_ms=float(np.percentile(times,95)),windows_per_second=1000/median,times_ms=times))
  if hasattr(q,'close'):q.close()
  del q;gc.collect()
  if a.backend=='mlx':mx.clear_cache()
result=dict(host=platform.node(),model=a.model,backend=a.backend,platform=platform.platform(),scope='single raw host input to host probabilities; internal float64 preprocessing/output allocation included; GPU synchronized including transfers; no NMS; 3 warmups then measured repeats',records=rows)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**result,'records':[{k:v for k,v in r.items() if k!='times_ms'} for r in rows]},indent=2))
