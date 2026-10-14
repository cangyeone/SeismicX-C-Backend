# SPDX-License-Identifier: GPL-3.0-only
"""Complete CPU raw batch to per-record min-heap picks; one actual run per precision."""
import argparse,json,sys,time,platform
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pnsn import Picker
p=argparse.ArgumentParser();p.add_argument('--model',choices=['pnsn','china-unet'],required=True);p.add_argument('--batch',type=int,default=4096);p.add_argument('--length',type=int,default=10240);p.add_argument('--output',required=True);a=p.parse_args()
x=np.random.default_rng(20261014).standard_normal((a.batch,3,a.length),dtype=np.float32);rows=[]
for mode in ['fp32','int8']:
 with Picker(ROOT/f'models/{a.model}-{mode}.bin',window=a.length,threads=1) as q:
  q.predict_batch(x[:1]);start=time.perf_counter();out=q.pick_batch(x);elapsed=time.perf_counter()-start
  assert len(out)==a.batch and all((v[:,1]>=0).all() and (v[:,1]<a.length).all() for v in out)
  rows.append(dict(model=a.model,backend='cpu',precision=mode,batch=a.batch,length=a.length,threads=1,stage='raw_host_to_independent_picks',seconds=elapsed,windows_per_second=a.batch/elapsed,picks=sum(len(v) for v in out),repeats=1))
result=dict(host=platform.node(),scope='one complete CPU batch run per precision; includes internal preprocessing and independent C min-heap NMS; synthetic sparse candidates',records=rows)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
