# SPDX-License-Identifier: GPL-3.0-only
"""Arbitrary-length end-to-end picking including heap NMS; raw+diff optional."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pnsn import Picker

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--mlx',action='store_true');p.add_argument('--cuda',action='store_true');p.add_argument('--legacy-jit');p.add_argument('--difference',action='store_true');p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',required=True);a=p.parse_args()
    x=np.load(a.input);records=[]
    factories={}
    for mode in ['fp32','int8']:
        factories['c-cpu-'+mode]=Picker(ROOT/f'models/pnsn-{mode}.bin',difference=int(a.difference))
        if a.cuda:factories['c-cuda-'+mode]=Picker(ROOT/f'models/pnsn-{mode}.bin',ROOT/'build/libpnsn_cuda.so',difference=int(a.difference))
        if a.mlx:
            from pnsn_mlx import MLXPicker
            factories['mlx-'+mode]=MLXPicker(ROOT/f'models/pnsn-{mode}.bin',difference=int(a.difference))
    if a.legacy_jit:
        if a.difference:raise ValueError('Compare raw traces with raw legacy JIT only')
        import torch
        torch.set_num_threads(1);model=torch.jit.load(a.legacy_jit).eval()
        def legacy(x):
            with torch.no_grad():return model(torch.from_numpy(x.astype(np.float32))).numpy()
        factories['upstream-jit-cpu']=legacy
    for length in [len(x),len(x)*10]:
        wave=np.tile(x,(length//len(x),1))
        for name,backend in factories.items():
            fn=backend.pick if hasattr(backend,'pick') else backend
            begin=time.perf_counter();out=fn(wave);cold=(time.perf_counter()-begin)*1000;times=[]
            for _ in range(a.repeats):
                begin=time.perf_counter();out=fn(wave);times.append((time.perf_counter()-begin)*1000)
            assert np.isfinite(out).all()
            records.append(dict(backend=name,length=length,picks=len(out),cold_ms=cold,median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95))))
    result=dict(input=a.input,difference=a.difference,repeats=a.repeats,records=records,
                comparison='Legacy uses float32 mean, clamps indices to N-2 and creates ceil(N/9216) windows. Native uses float64 mean, pads by final sample, stops once covered and clips to true length. Timing comparison is not an exact equivalence test.')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
