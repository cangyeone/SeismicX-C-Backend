# SPDX-License-Identifier: GPL-3.0-only
"""True PyTorch CUDA batch benchmark; input/output already resident on the GPU."""
import argparse,json,sys,time,gc,platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from export import load_network
p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=4096);p.add_argument('--length',type=int,default=10240);p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
torch.manual_seed(20261014);model=load_network(ROOT.parent/'ckpt/pnsn.v3.pt').cuda().eval()
# Distinct seeded records in every batch slot; no repeated identical window shortcuts.
x=torch.randn(a.batch,3,a.length,device='cuda',dtype=torch.float32)
records=[]
def measure(name,fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    begin=time.perf_counter();y=fn();torch.cuda.synchronize();cold=time.perf_counter()-begin
    assert y.shape==(a.batch,5,a.length) and torch.isfinite(y).all().item()
    del y
    for _ in range(2):y=fn();torch.cuda.synchronize();del y
    for i in range(a.repeats):
        begin=time.perf_counter();y=fn();torch.cuda.synchronize();elapsed=time.perf_counter()-begin;times.append(elapsed);del y
        print(json.dumps(dict(stage=name,iteration=i,seconds=elapsed)),flush=True)
    sec=float(np.median(times));records.append(dict(backend='torch-cuda',precision='fp32',stage=name,batch=a.batch,length=a.length,cold_seconds=cold,seconds=times,median_seconds=sec,p95_seconds=float(np.percentile(times,95)),windows_per_second=a.batch/sec,signal_hours_per_second=a.batch*a.length/100/3600/sec,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
with torch.inference_mode():
    try:
        measure('network_gpu_resident',lambda:model(x))
        torch.cuda.empty_cache()
        def preprocess_forward():
            shifted=x.double()-x[:,:,:1].double();centered=shifted-shifted.mean(-1,keepdim=True)
            normalized=(centered/(centered.std(-1,correction=1,keepdim=True)+1e-6)).float()
            return model(normalized)
        measure('float64_preprocess_and_network_gpu_resident',preprocess_forward)
        # Equal end-to-end host boundary and seed as the native CUDA benchmark.
        del x
        torch.cuda.empty_cache()
        host=np.random.default_rng(20261014).standard_normal((a.batch,3,a.length),dtype=np.float32)
        def host_forward():
            raw=torch.from_numpy(host).cuda()
            shifted=raw.double()-raw[:,:,:1].double();centered=shifted-shifted.mean(-1,keepdim=True)
            normalized=(centered/(centered.std(-1,correction=1,keepdim=True)+1e-6)).float()
            return model(normalized).cpu()
        measure('raw_host_to_host_probabilities',host_forward)
    except torch.cuda.OutOfMemoryError as e:records.append(dict(status='out_of_memory',message=str(e)))
result=dict(host=platform.node(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,batch=a.batch,length=a.length,repeats=a.repeats,tf32=False,input='distinct seeded float32 records [B,3,T] on GPU; 100 Hz interpretation',timing='synchronized wall clock; excludes generation/model loading/NMS; resident stages exclude transfers, host stage includes H2D/D2H and GPU float64 normalization',records=records)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
