"""Numerical, boundary, preprocessing and exact NMS regression checks."""
import argparse, ctypes as ct, json, sys, time, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tools'))
from pnsn import Picker, Pick, load_library

def nms_reference(candidates,gap):
    selected=[]
    for phase,sample,confidence in sorted(candidates,key=lambda p:(-p[2],p[1],p[0])):
        if not any(phase==a and abs(sample-b)<=gap for a,b,_ in selected):selected.append((phase,sample,confidence))
    return sorted(selected,key=lambda p:(p[1],p[0]))

def normalize(x):
    x=np.asarray(x,dtype=np.float64);x=x-x[0];x=x-x.mean(axis=0)
    return (x/(x.std(axis=0,ddof=1)+1e-6)).astype(np.float32)

def picks(prob,lib,gap=300,threshold=.1):
    t,p=np.where(prob[:,1:]>threshold);n=len(t)
    c=(Pick*n)(*(Pick(int(phase),int(at),float(prob[at,phase+1])) for at,phase in zip(t,p)))
    out=(Pick*n)();count=ct.c_size_t()
    if lib.pnsn_nms(c,n,gap,out,n,ct.byref(count)):raise RuntimeError(lib.pnsn_last_error().decode())
    return [(out[i].phase,out[i].sample,out[i].confidence) for i in range(count.value)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library');p.add_argument('--mlx',action='store_true');p.add_argument('--output',default=str(ROOT/'reports/validation.json'));a=p.parse_args()
    lib=load_library(a.library);rng=np.random.default_rng(20261014);tests=[];numeric=[]
    import torch
    from export import load_network
    torch.set_num_threads(1);network=load_network(ROOT.parent/'ckpt/pnsn.v3.pt')
    for trial in range(100):
        gap=int(rng.integers(0,20));candidates=[(int(rng.integers(0,4)),int(rng.integers(0,100)),float(np.float32(rng.integers(0,10)/10))) for _ in range(200)]
        n=len(candidates);c=(Pick*n)(*(Pick(*x) for x in candidates));out=(Pick*n)();count=ct.c_size_t()
        assert not lib.pnsn_nms(c,n,gap,out,n,ct.byref(count))
        got=[(out[i].phase,out[i].sample,out[i].confidence) for i in range(count.value)]
        assert got==nms_reference(candidates,gap),(trial,got)
    tests.append('100 randomized exact-NMS comparisons with duplicate times and tied scores')
    for length in (128,6144,10240):
        x=rng.normal(size=(length,3))+1e12
        with torch.no_grad():ref=network(torch.from_numpy(normalize(x).T.copy())[None])[0].T.numpy()
        for mode in ('fp32','int8'):
            with Picker(ROOT/f'models/pnsn-{mode}.bin',a.library,window=length,overlap=min(1024,length//4)) as picker:
                out=picker.predict(x)
                error=np.abs(out-ref)
                assert np.isfinite(out).all() and np.max(abs(out.sum(1)-1))<2e-6
                if mode=='fp32':assert error.max()<2e-5
                else:assert error.max()<.25 and error.mean()<.01 # drift smoke gate, not a scientific accuracy gate
                numeric.append(dict(backend='c',precision=mode,length=length,max_error=float(error.max()),mean_error=float(error.mean())))
                if a.mlx:
                    from pnsn_mlx import MLXPicker
                    mlx=MLXPicker(ROOT/f'models/pnsn-{mode}.bin',window=length,overlap=min(1024,length//4))
                    y=mlx.predict(x);e=abs(y-ref)
                    if mode=='fp32':assert e.max()<2e-5
                    else:assert e.max()<.25 and e.mean()<.01
                    numeric.append(dict(backend='mlx',precision=mode,length=length,max_error=float(e.max()),mean_error=float(e.mean())))
    tests.append('BN fusion, fixed windows 128/6144/10240 and probability simplex')
    with Picker(ROOT/'models/pnsn-fp32.bin',a.library,window=128,overlap=32,threshold=.3,min_gap=7) as picker:
        for dtype in (np.float32,np.float64,np.int32):
            x=np.full((128,3),2e9,dtype=dtype);out=picker.predict(x)
            assert np.isfinite(out).all()
        x=rng.integers(np.iinfo(np.int32).min,np.iinfo(np.int32).max,size=(128,3),dtype=np.int32)
        assert np.max(abs(picker.predict(x)-picker.predict(x.astype(np.float64))))<1e-7
        for length in (0,1,2,127,128,129,223,224,225,513):
            x=rng.normal(size=(length,3));actual=picker.pick(x)
            candidates=[]
            for start in range(0,max(length,1),96):
                if not length:break
                window=np.pad(x[start:start+128],((0,max(0,128-(length-start))),(0,0)),mode='edge')
                prob=picker.predict(window)
                at,phase=np.where(prob[:min(128,length-start),1:]>.3)
                candidates.extend((int(ph),int(t+start),float(prob[t,ph+1])) for t,ph in zip(at,phase))
                if length-start<=128:break
            reference=nms_reference(candidates,7)
            assert actual.tolist()==[list(v) for v in reference]
        tests.append('arbitrary length 0/1/2/tail boundaries; no picks outside true length')
        for bad in (np.nan,np.inf,-np.inf):
            x=np.ones((128,3));x[33,0]=bad
            try:picker.predict(x);raise AssertionError('accepted nonfinite')
            except RuntimeError:pass
        try:picker.predict(np.ones((127,3)));raise AssertionError('accepted wrong fixed length')
        except RuntimeError:pass
    tests.append('constant channels, int32 extrema, nonfinite rejection and huge float64 DC offsets')
    with Picker(ROOT/'models/pnsn-fp32.bin',a.library,window=128,overlap=32,threshold=0,max_candidates=1) as picker:
        try:picker.pick(np.zeros((128,3)));raise AssertionError('silently exceeded budget')
        except RuntimeError as e:assert 'budget' in str(e)
    tests.append('candidate memory limit fails explicitly')
    # Real high-amplitude convolution/LSTM states exercise the Metal saturation fix.
    real=ROOT/'reports/real-waveform.npy'
    if real.exists():
        raw=np.load(real)
        with Picker(ROOT/'models/pnsn-fp32.bin',a.library) as native:
            reference=native.predict(raw[:10240]);full=native.pick(raw)
            assert len(full)>0 and (full[:,1]>=0).all() and (full[:,1]<len(raw)).all()
            if a.mlx:
                for mode in ('fp32','int8'):
                    mlx=MLXPicker(ROOT/f'models/pnsn-{mode}.bin')
                    for repeat in range(3):
                        y=mlx.predict(raw[:10240]);assert np.isfinite(y).all()
                        if mode=='fp32':assert np.max(abs(y-reference))<2e-5
                    result=mlx.pick(raw);assert len(result)>0
                tests.append('repeated real-waveform MLX fixed/long picks, high-amplitude Metal states')
        tests.append('real three-component waveform, full 26001-sample trace')
    with Picker(ROOT/'models/pnsn-fp32.bin',a.library,window=128,overlap=32,threshold=.3,min_gap=7,difference=1) as native:
        raw=rng.normal(size=(225,3));combined=[]
        for start in (0,96,192):
            for diff in (0,1):
                normalized=np.empty((128,3),np.float32)
                assert not lib.pnsn_preprocess(raw.ctypes.data,1,len(raw),start,128,diff,normalized.ctypes.data)
                with torch.no_grad():prob=network(torch.from_numpy(normalized.T.copy())[None])[0].T.numpy()
                at,phase=np.where(prob[:min(128,len(raw)-start),1:]>.3)
                combined.extend((int(ph),int(t+start),float(prob[t,ph+1])) for t,ph in zip(at,phase))
            if len(raw)-start<=128:break
        expected=nms_reference(combined,7);actual=native.pick(raw)
        assert [(int(p),int(t)) for p,t,s in actual]==[(p,t) for p,t,s in expected]
    tests.append('raw plus first-difference ensemble with shared heap suppression')
    data=(ROOT/'models/pnsn-int8.bin').read_bytes()
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'bad.bin'
        for corrupted in (b'',data[:16],b'BADMAGIC'+data[8:],data[:-1],data+b'junk',data[:20]+b'\xff'*4+data[24:]):
            path.write_bytes(corrupted)
            try:Picker(path,a.library);raise AssertionError('accepted corrupt container')
            except RuntimeError:pass
    tests.append('strict truncated/corrupted/trailing model rejection')
    result=dict(status='passed',cpu_features=lib.pnsn_cpu_features().decode(),tests=tests,numerical=numeric)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
