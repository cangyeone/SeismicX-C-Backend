"""Paired v3/native INT8 evaluation on immutable existing held-out Pg/Sg windows.
Cache format: test.npy [N,variants,3,10240], test_labels.npy [N,variants,4].
No cache writes, model training, or source experiment mutations.
"""
import argparse, json, sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'tools'))
from pnsn import Picker,load_library
from validate import normalize,picks
from export import load_network

def rates(count):
    tp,fp,fn=map(int,count)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)

def counts(pred,labels):
    result=np.zeros((2,3),np.int64)
    for phase in (0,1):
        arrivals=[int(t) for p,t,s in pred if p==phase];truth=int(labels[phase])
        hit=truth>=0 and any(abs(t-truth)<=50 for t in arrivals)
        result[phase]=[int(hit),len(arrivals)-int(hit),int(truth>=0 and not hit)]
    return result

def paired(reference,got):
    remaining=list(got);delta=[];confidence=[];matched=0
    for phase,at,score in reference:
        nearby=[q for q in remaining if q[0]==phase and abs(q[1]-at)<=50]
        if nearby:
            best=min(nearby,key=lambda q:abs(q[1]-at));remaining.remove(best);delta.append(abs(best[1]-at));confidence.append(abs(best[2]-score));matched+=1
    return matched,delta,confidence

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=['pnsn','china-unet'],default='pnsn');parser.add_argument('--cache',required=True);parser.add_argument('--per-province',type=int,default=0,help='0 means all test windows');parser.add_argument('--output',required=True);parser.add_argument('--backend',choices=['cpu','cuda','cudnn','cudnn_int8'],default='cuda');a=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1);model=(load_network(ROOT.parent/'ckpt/pnsn.v3.pt') if a.model=='pnsn' else __import__('export_unet').load_network(ROOT.parent/'ckpt/china.unet.pt')).cuda(0);cpu_lib=load_library()
    library=str(ROOT/f'build/libpnsn_{a.backend}.so') if a.backend in ('cuda','cudnn','cudnn_int8') else None
    native={mode:Picker(ROOT/f'models/{a.model}-{mode}.bin',library) for mode in ('fp32','int8')}
    totals={key:np.zeros((2,3),np.int64) for key in ['torch-fp32','native-fp32','native-int8']}
    comparisons={mode:dict(reference=0,picks=0,matched=0,deltas=[],score_errors=[],max_probability_error=0,absolute_error_sum=0,probabilities=0) for mode in native}
    rows=[];started=time.perf_counter();windows=0;rng=np.random.default_rng(20261014)
    for province in sorted(Path(a.cache).iterdir()):
        path=province/'test.npy'
        if not path.exists():continue
        waves=np.load(path,mmap_mode='r');labels=np.load(province/'test_labels.npy',mmap_mode='r')
        metadata=json.loads((province/"test.json").read_text())
        if len(metadata)>len(waves) or len(labels)!=len(waves):raise ValueError("Inconsistent held-out cache")
        indices=np.arange(len(metadata))
        if a.per_province and len(indices)>a.per_province:indices=np.sort(rng.choice(indices,a.per_province,replace=False))
        province_counts={key:np.zeros((2,3),np.int64) for key in totals}
        for begin in range(0,len(indices),16):
            batch=indices[begin:begin+16]
            # Cached waves are already normalized; normalize again identically for all runtimes.
            raw=[np.asarray(waves[index,0].T,dtype=np.float64) for index in batch]
            x=np.stack([normalize(w).T for w in raw])
            reference=model(torch.from_numpy(x).cuda(0)).cpu().numpy().transpose(0,2,1)
            for at,(index,w,ref) in enumerate(zip(batch,raw,reference)):
                pred=picks(ref,cpu_lib)
                truth=np.asarray(labels[index,0]);c=counts(pred,truth);totals['torch-fp32']+=c;province_counts['torch-fp32']+=c
                for mode,picker in native.items():
                    out=picker.predict(w);prediction=picks(out,picker.lib);c=counts(prediction,truth)
                    key='native-'+mode;totals[key]+=c;province_counts[key]+=c
                    compare=comparisons[mode];matched,delta,score=paired(pred,prediction)
                    compare['reference']+=len(pred);compare['picks']+=len(prediction);compare['matched']+=matched
                    compare['deltas'].extend(delta);compare['score_errors'].extend(score)
                    error=abs(out-ref);compare['max_probability_error']=max(compare['max_probability_error'],float(error.max()))
                    compare['absolute_error_sum']+=float(error.sum(dtype=np.float64));compare['probabilities']+=out.size
                windows+=1
        row=dict(province=province.name,windows=len(indices),metrics={key:rates(value.sum(0)) for key,value in province_counts.items()});rows.append(row)
        print(json.dumps(dict(province=province.name,windows_total=windows,elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    metrics={key:dict(micro=rates(value.sum(0)),phases={p:rates(value[i]) for i,p in enumerate(['Pg','Sg'])}) for key,value in totals.items()}
    for mode,c in comparisons.items():
        d=c.pop('deltas');s=c.pop('score_errors');c['matched_fraction']=c['matched']/c['reference'] if c['reference'] else None
        c['unmatched_reference']=c['reference']-c['matched'];c['unmatched_native']=c['picks']-c['matched']
        c['sample_error_median']=float(np.median(d)) if d else None;c['sample_error_p95']=float(np.percentile(d,95)) if d else None;c['sample_error_max']=max(d,default=None)
        c['confidence_error_max']=max(s,default=None);c['mean_probability_error']=c.pop('absolute_error_sum')/c.pop('probabilities')
    result=dict(windows=windows,provinces=len(rows),seed=20261014,backend=a.backend,int8_compute=(('all convolution W8A8' if a.model=='china-unet' else 'conv W8A8 / FP32 LSTM') if a.backend=='cudnn_int8' else 'FP32 dequantized Q8 weights' if a.backend=='cudnn' else 'W8A8'),gpu=torch.cuda.get_device_name(0),
                model=a.model,scope=('Held-out 2020-2022 test cache; manual Pg/Sg arrivals used for China UNet P/S outputs' if a.model=='china-unet' else 'Held-out 2020-2022 cached test windows from existing 2019 pilot; labels evaluated for Pg/Sg only; all four predicted phases compared against FP32, not manual Pn/Sn accuracy'),
                input='Existing normalized variant 0, limited to actual test.json row count (exclude padded cache slots), normalized identically again in float64; TF32 disabled; no retraining',
                threshold=.1,nms_gap=300,match_tolerance_samples=50,metrics=metrics,paired=comparisons,rows=rows,elapsed_s=time.perf_counter()-started)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for p in native.values():p.close()
if __name__=='__main__':main()
