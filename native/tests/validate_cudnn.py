"""Vendor cuDNN vs explicit PyTorch graph, including Q8-storage/FP32 compute."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests'),str(ROOT/'tools')]
from pnsn import Picker
from format import read_model
from validate import normalize
from export import load_network
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--int8',action='store_true');a=p.parse_args()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(1)
def graph(path,integer=False):
 bits,ls=read_model(path)
 weights=[];qweights=[];scales=[]
 for l in ls:
  raw=l['w'].astype(np.float32); q=raw[:,:l['ni']*l['k']].reshape(l['no'],l['k'],l['ni']).transpose(0,2,1).copy()
  if l['kind']==2:q=q.transpose(1,0,2).copy()
  qweights.append(torch.from_numpy(q).cuda());scales.append(torch.from_numpy(l['scale']).cuda()[None,:,None])
  w=raw.copy()
  if bits==8:w*=l['scale'][:,None]
  w=w[:,:l['ni']*l['k']].reshape(l['no'],l['k'],l['ni']).transpose(0,2,1).copy()
  if l['kind']==2:w=w.transpose(1,0,2).copy()
  weights.append((torch.from_numpy(w).cuda(),torch.from_numpy(l['bias']).cuda()))
 rnns=[]
 for base in [7,12]:
  r=torch.nn.LSTM(ls[base]['ni'],96,bidirectional=True,batch_first=True).cuda().eval()
  for d,suffix in enumerate(['','_reverse']):
   for q,kind in enumerate(['ih','hh']):
    w,b=weights[base+2*d+q]
    getattr(r,'weight_'+kind+'_l0'+suffix).data.copy_(w[:,:,0]);getattr(r,'bias_'+kind+'_l0'+suffix).data.copy_(b)
  r.flatten_parameters();rnns.append(r)
 def conv(x,i):
  l=ls[i]
  if integer and bits==8:
   scale=torch.clamp(x.abs().amax((1,2),keepdim=True)/127,min=1e-30)
   q=torch.clamp(torch.round(x/scale),-127,127)
   if l['kind']==2:y=F.conv_transpose1d(q,qweights[i],stride=2,padding=l['pad'],output_padding=1)
   else:y=F.conv1d(q,qweights[i],padding=l['pad'])
   return y*scale*scales[i]+weights[i][1][None,:,None]
  if l['kind']==2:return F.conv_transpose1d(x,*weights[i],stride=2,padding=l['pad'],output_padding=1)
  return F.conv1d(x,*weights[i],padding=l['pad'])
 def run(x):
  with torch.no_grad():
   for i in range(7):x=F.max_pool1d(F.relu(conv(x,i)),2)
   for q,r in enumerate(rnns):x=conv(r(x.transpose(1,2))[0].transpose(1,2),11+q*5)
   for i in range(17,24):x=F.relu(conv(x,i))
   return F.softmax(conv(x,24),dim=1).cpu().numpy()
 return run
rng=np.random.default_rng(20261014);rows=[]
for mode in ['fp32','int8']:
 path=ROOT/f'models/pnsn-{mode}.bin';ref=graph(path,integer=a.int8)
 for length in [128,6144,10240]:
  with Picker(path,ROOT/('build/libpnsn_cudnn_int8.so' if a.int8 else 'build/libpnsn_cudnn.so'),window=length,overlap=length//8) as picker:
   for b in [1,3]:
    raw=rng.normal(size=(b,3,length)).astype(np.float64)+1e12
    normalized=np.stack([normalize(x.T).T for x in raw]);expected=ref(torch.from_numpy(normalized).cuda())
    got=picker.predict_batch(raw);err=float(abs(got-expected).max())
    expert=picker.predict_batch(normalized,normalized=True);expert_error=float(abs(expert-expected).max())
    print(dict(mode=mode,batch=b,length=length,raw_error=err,expert_error=expert_error,preprocessing_drift=float(abs(expert-got).max())),flush=True)
    assert expert_error<(0.03 if a.int8 and mode=='int8' else 3e-5),(mode,b,length,expert_error)
    assert err<(0.03 if a.int8 and mode=='int8' else 3e-5),(mode,b,length,err)
    rows.append(dict(oracle_max_probability_error=expert_error,preprocessing_probability_drift=float(abs(expert-got).max()),model=mode,compute=('conv W8A8 / LSTM FP32' if a.int8 and mode=='int8' else 'FP32'),length=length,batch=b,max_probability_error=err))
 real=ROOT/'reports/real-waveform.npy'
 if real.exists():
  waves=np.load(real);x=np.stack([waves[:10240].T,waves[5000:15240].T,waves[9000:19240].T])
  with Picker(path,ROOT/('build/libpnsn_cudnn_int8.so' if a.int8 else 'build/libpnsn_cudnn.so')) as picker:
   norm=np.stack([normalize(w.T).T for w in x]);expected=ref(torch.from_numpy(norm).cuda());got=picker.predict_batch(x)
   err=float(abs(got-expected).max());assert err<(0.03 if a.int8 and mode=='int8' else 5e-5),(mode,'real',err)
   rows.append(dict(model=mode,compute=('conv W8A8 / LSTM FP32' if a.int8 and mode=='int8' else 'FP32'),input='public real waveform',batch=3,max_probability_error=err))
 original=load_network(ROOT.parent/'ckpt/pnsn.v3.pt').cuda().eval()
 x=rng.normal(size=(3,3,10240)).astype(np.float32);norm=np.stack([normalize(w.T).T for w in x])
 with Picker(path,ROOT/('build/libpnsn_cudnn_int8.so' if a.int8 else 'build/libpnsn_cudnn.so')) as picker,torch.no_grad():
  expected=original(torch.from_numpy(norm).cuda()).cpu().numpy();got=picker.predict_batch(x);err=float(abs(got-expected).max())
  if mode=='fp32':assert err<3e-5
  rows.append(dict(model=mode,reference='original unfused checkpoint',batch=3,max_probability_error=err))
result=dict(status='passed',gpu=torch.cuda.get_device_name(0),cudnn_version=torch.backends.cudnn.version(),records=rows)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
