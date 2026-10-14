"""4096 independent realtime [B,3,L] records, true native CUDA batch or CPU fallback."""
import argparse,gc,json,sys,time,platform
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pnsn import Picker
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','cuda','cudnn','cudnn_int8','mlx'],default='cuda');p.add_argument('--batch',type=int,default=4096);p.add_argument('--length',type=int,default=10240);p.add_argument('--repeats',type=int,default=5);p.add_argument('--output',required=True);p.add_argument('--post',action='store_true');p.add_argument('--model',choices=['pnsn','china-unet'],default='pnsn');p.add_argument('--precision',choices=['both','fp32','int8'],default='both');a=p.parse_args()
x=np.random.default_rng(20261014).standard_normal((a.batch,3,a.length),dtype=np.float32)
records=[]
for precision in (['fp32','int8'] if a.precision=='both' else [a.precision]):
 if a.backend=='mlx':
  import mlx.core as mx
  from pnsn_mlx import MLXPicker
  picker=MLXPicker(ROOT/f'models/{a.model}-{precision}.bin',window=a.length);mx.reset_peak_memory()
 else:picker=Picker(ROOT/f'models/{a.model}-{precision}.bin',ROOT/f'build/libpnsn_{a.backend}.so' if a.backend in ('cuda','cudnn','cudnn_int8') else None,window=a.length)
 times=[];gpu_times=[];prep_times=[];samples=[]
 for iteration in range(a.repeats+1):
  begin=time.perf_counter();y=picker.predict_batch(x);elapsed=time.perf_counter()-begin
  assert y.shape==(a.batch,3 if a.model=='china-unet' else 5,a.length) and np.isfinite(y).all()
  assert np.max(abs(y.sum(1)-1))<2e-6
  if iteration==0:
   cold=elapsed;subset=x[[0,a.batch//2,a.batch-1]]
   single=np.stack([picker.predict(w.T).T for w in subset]);error=float(np.max(abs(y[[0,a.batch//2,a.batch-1]]-single)))
   assert error<(2e-5 if precision=='fp32' else .03),error
  else:
   times.append(elapsed)
   if a.backend in ('cuda','cudnn','cudnn_int8'):
    stats=picker.batch_device_stats();gpu_times.append(stats['network_ms']/1000);prep_times.append(stats['preprocess_ms']/1000)
  print(json.dumps(dict(backend=a.backend,precision=precision,iteration=iteration,seconds=elapsed)),flush=True)
  del y;gc.collect()
 median=float(np.median(times));row=dict(backend=a.backend,precision=precision,compute=(('all_conv_w8a8' if a.model=='china-unet' else 'conv_w8a8_lstm_fp32') if a.backend=='cudnn_int8' and precision=='int8' else 'fp32_dequantized_weights' if a.backend=='cudnn' and precision=='int8' else 'W8A32' if a.backend=='mlx' and precision=='int8' else precision),batch=a.batch,length=a.length,cold_seconds=cold,seconds=times,median_seconds=median,p95_seconds=float(np.percentile(times,95)),windows_per_second=a.batch/median,signal_hours_per_second=a.batch*a.length/100/3600/median,single_vs_batch_max_error=error,
  scope='raw host input to host probabilities, includes preprocessing, GPU H2D/D2H (CUDA), allocation, no NMS')
 if a.backend in ('cuda','cudnn','cudnn_int8'):row.update(network_gpu_seconds=gpu_times,network_gpu_median_seconds=float(np.median(gpu_times)),preprocess_gpu_seconds=prep_times,preprocess_gpu_median_seconds=float(np.median(prep_times)),**picker.batch_device_stats())
 if a.backend=='mlx':row.update(peak_mlx_active_bytes=mx.get_peak_memory(),mlx_version=mx.__version__,batch_execution=('layerwise im2col tiles of 128 records, skip connections, no recurrence' if a.model=='china-unet' else 'layerwise im2col tiles of 128 records, full B parallel Metal recurrent blocks'))
 if a.backend=='cpu':row.update(batch_execution='SIMD CPU fallback processes records serially',cpu_features=picker.lib.pnsn_cpu_features().decode(),threads=1)
 records.append(row)
 if a.post:
  begin=time.perf_counter();out=picker.pick_batch(x);elapsed=time.perf_counter()-begin
  assert len(out)==a.batch and all((v[:,1]>=0).all() and (v[:,1]<a.length).all() for v in out)
  records.append(dict(backend=a.backend,precision=precision,compute=(('all_conv_w8a8' if a.model=='china-unet' else 'conv_w8a8_lstm_fp32') if a.backend=='cudnn_int8' and precision=='int8' else 'fp32_dequantized_weights' if a.backend=='cudnn' and precision=='int8' else 'W8A32' if a.backend=='mlx' and precision=='int8' else precision),batch=a.batch,length=a.length,stage='raw_host_to_independent_picks',seconds=elapsed,windows_per_second=a.batch/elapsed,picks=sum(len(v) for v in out),repeats=1))
 if hasattr(picker,'close'):picker.close()
 del picker;gc.collect()
 if a.backend=='mlx':mx.clear_cache()
result=dict(host=platform.node(),platform=platform.platform(),batch=a.batch,length=a.length,seed=20261014,model=a.model,input=f'{a.batch} distinct seeded float32 records, [B,3,L]',output=f'probabilities [B,{3 if a.model=="china-unet" else 5},L], picks independently NMS per record',timing='synchronized wall clock; generation/model loading/validation excluded; first run cold, then measured repeats; CPU serial fallback clearly identified',records=records)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
