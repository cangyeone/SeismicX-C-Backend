"""Two official strategies: history [3,N] and realtime [B,3,L]."""
import argparse,ctypes as ct,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pnsn import Picker,Pick
p=argparse.ArgumentParser();p.add_argument('--library');p.add_argument('--mlx',action='store_true');p.add_argument('--output',default=str(ROOT/'reports/batch-validation.json'));a=p.parse_args()
rng=np.random.default_rng(20261014);records=[]
for length in [128,6144,10240]:
 for mode in ['fp32','int8']:
  with Picker(ROOT/f'models/pnsn-{mode}.bin',a.library,window=length,overlap=length//8,threshold=.1) as picker:
   for b in [1,3,5]:
    x=rng.normal(size=(b,3,length)).astype(np.float64)+1e12
    y=picker.predict_batch(x);reference=np.stack([picker.predict(w.T).T for w in x])
    error=float(np.max(abs(y-reference)));assert error<(2e-5 if mode=='fp32' else .03),(mode,length,b,error)
    selected=picker.pick_batch(x)
    for i in range(b):
     history=picker.pick_history(x[i]);assert np.array_equal(history[:,:2],selected[i][:,:2]),(mode,length,b,i,history,selected[i])
    records.append(dict(precision=mode,length=length,batch=b,max_error=error,device_stats=picker.batch_device_stats()))
   x=rng.integers(-2147483648,2147483647,size=(3,3,length),dtype=np.int32)
   assert np.max(abs(picker.predict_batch(x)-picker.predict_batch(x.astype(np.float64))))<1e-6
   assert picker.predict_batch(np.empty((0,3,length))).shape==(0,5,length)
   assert picker.pick_batch(np.empty((0,3,length)))==[]
   assert picker.pick_history(np.empty((3,0))).shape==(0,3)
   for n in (1,2,length+3):
    h=rng.normal(size=(3,n));assert np.array_equal(picker.pick_history(h),picker.pick(h.T))
   bad=np.ones((2,3,length));bad[1,1,44]=np.nan
   try:picker.predict_batch(bad);raise AssertionError('Accepted NaN batch')
   except RuntimeError:pass
   try:picker.predict_batch(np.zeros((2,3,length+1)));raise AssertionError('Accepted wrong L')
   except ValueError:pass
   if a.mlx:
    from pnsn_mlx import MLXPicker
    mlx=MLXPicker(ROOT/f'models/pnsn-{mode}.bin',window=length,overlap=length//8)
    raw=rng.normal(size=(3,3,length))
    out=mlx.predict_batch(raw);ref=np.stack([mlx.predict(w.T).T for w in raw]);assert np.max(abs(out-ref))<2e-5
    assert mlx.pick_batch(np.empty((0,3,length)))==[]
# Real signals and batch permutations expose accidental state/mean/NMS sharing across B.
real=ROOT/'reports/real-waveform.npy'
if real.exists():
 x=np.load(real);waves=np.stack([x[:10240].T,x[5000:15240].T,np.zeros((3,10240)),x[9000:19240].T])
 for mode in ['fp32','int8']:
  with Picker(ROOT/f'models/pnsn-{mode}.bin',a.library) as native:
   y=native.predict_batch(waves);ref=np.stack([native.predict(w.T).T for w in waves]);assert np.max(abs(y-ref))<(2e-5 if mode=='fp32' else .03)
   permutation=[3,1,0,2];assert np.max(abs(native.predict_batch(waves[permutation])-y[permutation]))<2e-6
   selected=native.pick_batch(waves)
   for i in range(len(waves)):assert np.array_equal(selected[i][:,:2],native.pick_history(waves[i])[:,:2])
   rawdiff=Picker(ROOT/f'models/pnsn-{mode}.bin',a.library,difference=1)
   results=rawdiff.pick_batch(waves)
   for i in range(len(waves)):assert np.array_equal(results[i][:,:2],rawdiff.pick_history(waves[i])[:,:2])
   rawdiff.close()
   # Insufficient output capacity is atomic for both output and offsets.
   candidates=(Pick*1)();candidates[0].sample=777;bounds=(ct.c_size_t*(len(waves)+1))(*([777]*(len(waves)+1)));count=ct.c_size_t()
   if sum(len(v) for v in selected)>1:
    status=native.lib.pnsn_pick_batch(native.ctx,waves.ctypes.data,1,len(waves),10240,candidates,1,bounds,ct.byref(count))
    assert status==-2 and candidates[0].sample==777 and list(bounds)==[777]*5
   if a.mlx:
    mlx=MLXPicker(ROOT/f'models/pnsn-{mode}.bin');out=mlx.predict_batch(waves)
    ref=np.stack([mlx.predict(w.T).T for w in waves]);assert np.max(abs(out-ref))<2e-5
    for i,got in enumerate(mlx.pick_batch(waves)):assert np.array_equal(got[:,:2],mlx.pick_history(waves[i])[:,:2])
result=dict(status='passed',tests=['planar history vs legacy interleaved','batch 1/3/5, L 128/6144/10240 vs independent single calls','float64 1e12 DC and int32 extrema','batch empty/NaN/wrong shape','batch permutation and real-signal independence','independent raw+diff NMS and atomic output capacity'],records=records)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
