# SPDX-License-Identifier: GPL-3.0-only
"""Export the selected cangyeone China PhaseNet-style UNet, all 10 BN fused.
This is the nearest-neighbor upsampling architecture in models/UNet.py,
not the original AI4EPS PhaseNet or SeisBench PhaseNet architecture.
"""
import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from models.UNet import UNet

def load_network(checkpoint):
 model=UNet().eval();s=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if 'state_dict' in s:s=s['state_dict']
 model.load_state_dict({k.removeprefix('model.'):v for k,v in s.items()},strict=True);return model

def layers(model):
 result=[]
 for block in [model.inputs]+[getattr(model,'layer'+str(i)) for i in range(10)]:
  up=hasattr(block,'layers') and isinstance(block.layers[0],torch.nn.UpsamplingNearest2d)
  if up:block=block.layers[1]
  if hasattr(block,'layers'):c,bn=block.layers[:2]
  else:c,bn=block,None
  w=c.weight.detach().double()[:,:,:,0];b=c.bias.detach().double()
  if bn is not None:
   scale=bn.weight.detach().double()/torch.sqrt(bn.running_var.double()+bn.eps)
   w*=scale[:,None,None];b=(b-bn.running_mean.double())*scale+bn.bias.detach().double()
  w=w.permute(0,2,1).reshape(c.out_channels,-1).float().numpy()
  result.append((3 if up else 1,c.in_channels,c.out_channels,c.kernel_size[0],4 if up else c.stride[0],c.padding[0],w,b.float().numpy()))
 return result

def export(checkpoint,output,precision):
 records=layers(load_network(checkpoint));out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('wb') as f:
  f.write(b'SXUNET1\0'+struct.pack('<II',11,8 if precision=='int8' else 32))
  for kind,ni,no,k,stride,pad,w,b in records:
   width=(ni*k+15)//16*16;weights=np.zeros((no,width),np.float32);weights[:,:ni*k]=w
   scale=np.maximum(np.abs(weights).max(1)/127,np.finfo(np.float32).tiny)
   q=np.clip(np.rint(weights/scale[:,None]),-127,127).astype(np.int8) if precision=='int8' else weights.astype('<f4')
   f.write(struct.pack('<7I',kind,ni,no,k,stride,pad,width));f.write(scale.astype('<f4').tobytes());f.write(b.astype('<f4').tobytes());f.write(q.tobytes())
 info=dict(format='SXUNET1',license='GPL-3.0-only',architecture='China PhaseNet-style UNet / nearest neighbor upsampling',precision=precision,layers=11,fused_bn=10,phases=['Noise','P','S'],window_multiple=256,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,quantization='symmetric per-output-channel weights; CPU dynamic per-patch W8A8; float nonlinearities')
 out.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',default=str(ROOT/'ckpt/china.unet.pt'));p.add_argument('--output',required=True);p.add_argument('--precision',choices=['fp32','int8'],default='int8');a=p.parse_args();export(a.checkpoint,a.output,a.precision)
