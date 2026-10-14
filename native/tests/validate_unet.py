# SPDX-License-Identifier: GPL-3.0-only
"""China PhaseNet-style UNet C FP32/INT8, history/batch/boundary regression."""
import argparse,ctypes as ct,json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tools'),str(ROOT/'tests')]
from pnsn import Picker,Pick
from export_unet import load_network
from validate import normalize,nms_reference
p=argparse.ArgumentParser();p.add_argument('--library');p.add_argument('--mlx',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(1);m=load_network(ROOT.parent/'ckpt/china.unet.pt');rng=np.random.default_rng(20261014);rows=[]
for length in [256,6144,10240]:
 x=rng.normal(size=(3,3,length)).astype(np.float64)+1e12
 norm=np.stack([normalize(w.T).T for w in x])
 with torch.no_grad():ref=m(torch.from_numpy(norm)).numpy()
 for mode in ['fp32','int8']:
  with Picker(ROOT/f'models/china-unet-{mode}.bin',a.library,window=length,overlap=length//4) as q:
   y=q.predict_batch(x);err=abs(y-ref);assert q.classes==3 and y.shape==ref.shape and np.isfinite(y).all()
   assert err.max()<(3e-5 if mode=='fp32' else .2) and err.mean()<.01,(mode,length,err.max(),err.mean())
   independent=np.stack([q.predict(w.T).T for w in x]);assert abs(y-independent).max()<(3e-5 if mode=='fp32' else .03)
   perm=[2,0,1];assert abs(q.predict_batch(x[perm])-y[perm]).max()<2e-5
   picks=q.pick_batch(x)
   for i,w in enumerate(x):assert np.array_equal(picks[i][:,:2],q.pick_history(w)[:,:2])
   extremes=rng.integers(-2147483648,2147483647,size=(2,3,length),dtype=np.int32);assert abs(q.predict_batch(extremes)-q.predict_batch(extremes.astype(np.float64))).max()<1e-6
   assert q.predict_batch(np.empty((0,3,length))).shape==(0,3,length) and q.pick_batch(np.empty((0,3,length)))==[]
   assert q.pick_history(np.empty((3,0))).shape==(0,3)
   for n in [1,2,length+3]:
    h=rng.normal(size=(3,n));assert np.array_equal(q.pick_history(h),q.pick(h.T))
   bad=x.copy();bad[1,0,0]=np.nan
   try:q.predict_batch(bad);raise AssertionError('Accepted NaN')
   except RuntimeError:pass
   if a.mlx:
    from pnsn_mlx import MLXPicker
    mx=MLXPicker(ROOT/f'models/china-unet-{mode}.bin',window=length,overlap=length//4);out=mx.predict_batch(x)
    assert abs(out-ref).max()<(3e-5 if mode=='fp32' else .2)
    assert mx.predict_batch(np.empty((0,3,length))).shape==(0,3,length)
   rows.append(dict(model=mode,length=length,batch=3,max_error=float(err.max()),mean_error=float(err.mean()),single_vs_batch_max_error=float(abs(y-independent).max())))
# Non-divisible batch tests the final cuDNN im2col tile/descriptors.
for mode in ['fp32','int8']:
 with Picker(ROOT/f'models/china-unet-{mode}.bin',a.library,window=256,overlap=64) as q:
  raw=rng.normal(size=(129,3,256)).astype(np.float32);out=q.predict_batch(raw)
  ids=[0,64,128];ref=np.stack([q.predict(raw[i].T).T for i in ids]);error=float(abs(out[ids]-ref).max())
  assert error<(3e-5 if mode=='fp32' else .03),(mode,'tile remainder',error)
  rows.append(dict(model=mode,length=256,batch=129,single_vs_batch_max_error=error))
real=ROOT/'reports/real-waveform.npy'
if real.exists():
 raw=np.load(real)
 with torch.no_grad():reference=m(torch.from_numpy(normalize(raw[:10240]).T.copy())[None])[0].T.numpy()
 for mode in ['fp32','int8']:
  with Picker(ROOT/f'models/china-unet-{mode}.bin',a.library,difference=1) as q:
   out=q.predict(raw[:10240]);assert np.isfinite(out).all();err=abs(out-reference)
   if mode=='fp32':assert err.max()<3e-5
   got=q.pick_history(raw.T);assert (got[:,1]>=0).all() and (got[:,1]<len(raw)).all()
   waves=np.stack([raw[:10240].T,raw[5000:15240].T,np.zeros((3,10240))]);selected=q.pick_batch(waves)
   for i,w in enumerate(waves):assert np.array_equal(selected[i][:,:2],q.pick_history(w)[:,:2])
   bounds=(ct.c_size_t*4)(777,777,777,777);small=(Pick*1)();small[0].sample=777;n=ct.c_size_t()
   if sum(len(v) for v in selected)>1:
    assert q.lib.pnsn_pick_batch(q.ctx,waves.ctypes.data,1,3,10240,small,1,bounds,ct.byref(n))==-2
    assert list(bounds)==[777]*4 and small[0].sample==777
   rows.append(dict(model=mode,input='real waveform',max_error=float(err.max()),mean_error=float(err.mean()),history_picks=len(got)))
for length in [128,6145]:
 try:Picker(ROOT/'models/china-unet-int8.bin',a.library,window=length,overlap=0);raise AssertionError('Accepted incompatible length')
 except RuntimeError:pass
result=dict(status='passed',architecture='china.unet.pt nearest-neighbor UNet, not standard PhaseNet',library=a.library or 'CPU',mlx=a.mlx,tests=['10 BN fusion','fixed 256/6144/10240','history arbitrary N=0/1/2/tail','batch 3, independent records/permutation/NMS','float64 DC/int32 extrema','empty/NaN/wrong window','real waveform/raw+diff/atomic capacity'],records=rows)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
