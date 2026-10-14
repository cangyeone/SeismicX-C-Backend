# SPDX-License-Identifier: GPL-3.0-only
"""Optional Apple GPU backend: MLX matmul/Q8 + custom Metal LSTM + shared C NMS.
Q8 uses the exact native int8 weights with floating point activations (W8A32).
"""
import ctypes as ct
import numpy as np
import mlx.core as mx
from pnsn import load_library, waveform, raw_array, Pick
from format import read_model

RECURRENCE = '''
    uint j = thread_position_in_threadgroup.x;
    uint batch = threadgroup_position_in_grid.x;
    threadgroup float hidden[96];
    threadgroup float cell[96];
    threadgroup float gates[384];
    if (j < 96) { hidden[j] = 0; cell[j] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int step = 0; step < STEPS; ++step) {
        int at = REVERSE ? STEPS - 1 - step : step;
        float sum = bias[j];
        for (int k = 0; k < 96; ++k) {
            float w = QUANT ? float(qweight[j * 96 + k]) * scales[j] : weight[j * 96 + k];
            sum += hidden[k] * w;
        }
        gates[j] = sum + inp[(batch * STEPS + at) * 384 + j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j < 96) {
            float i = 1.0f / (1.0f + metal::exp(-gates[j]));
            float f = 1.0f / (1.0f + metal::exp(-gates[96+j]));
            // Metal tanh must not overflow internally on large seismic gate values.
            float g = metal::tanh(metal::clamp(gates[192+j], -10.0f, 10.0f));
            float o = 1.0f / (1.0f + metal::exp(-gates[288+j]));
            cell[j] = f * cell[j] + i * g;
            hidden[j] = o * metal::tanh(metal::clamp(cell[j], -10.0f, 10.0f));
            out[(batch * STEPS + at) * 96 + j] = hidden[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
'''
class MLXPicker:
    def __init__(self,model,library=None,compile_graph=True,**options):
        if not mx.metal.is_available():raise RuntimeError('MLX Metal GPU unavailable')
        self.lib=load_library(library);self.options=self.lib.pnsn_default_options()
        for k,v in options.items():
            if k not in {n for n,_ in self.options._fields_}:raise ValueError(f'Unknown option {k}')
            setattr(self.options,k,v)
        o=self.options
        if o.window<128 or o.window>1048576 or o.window%128 or o.overlap>=o.window or o.min_gap<0 or not np.isfinite(o.threshold) or not 0<=o.threshold<=1 or not o.max_candidates or o.difference not in (0,1):raise ValueError('Invalid options')
        self.bits,records=read_model(model);self.layers=[];self.china=len(records)==11;self.classes=3 if self.china else 5
        if self.china and o.window%256:raise ValueError("China UNet requires a multiple of 256")
        for record in records:
            l=dict(record);w=record['w'];l['bias']=mx.array(record['bias']);l['scale']=mx.array(record['scale'])
            if self.bits==8:
                width=((w.shape[1]+63)//64)*64
                codes=np.full((record['no'],width),128,np.uint32)
                codes[:,:w.shape[1]]=w.astype(np.int32)+128
                u=codes.reshape(record['no'],-1,4)
                packed=u[:,:,0]|(u[:,:,1]<<8)|(u[:,:,2]<<16)|(u[:,:,3]<<24)
                scales=np.broadcast_to(record['scale'][:,None],(record['no'],width//64)).copy()
                l['packed']=mx.array(packed);l['qscales']=mx.array(scales);l['qbias']=mx.array(-128*scales)
                l['qweight']=mx.array(w);l['weight']=mx.zeros((1,),mx.float32);l['qwidth']=width
            else:l['weight']=mx.array(w);l['qweight']=mx.zeros((1,),mx.int8)
            self.layers.append(l)
        self.recur=mx.fast.metal_kernel(name='pnsn_lstm',input_names=['inp','weight','qweight','scales','bias'],output_names=['out'],source=RECURRENCE)
        self.indices={}
        t=o.window
        for index,l in enumerate(self.layers):
            if not self.china and index in (8,10,13,15):continue
            rows=t*4 if l['kind']==3 else t*2 if l['kind']==2 else (t+2*l['pad']-l['k'])//l['stride']+1
            if l['k']>1 or l['kind']==2:
                positions=np.arange(rows)[:,None]
                kernels=np.arange(l['k'])[None,:]
                if l['kind']==2:
                    source=positions+l['pad']-kernels;valid=source%2==0;source=source//2
                elif l['kind']==3:
                    source=positions+kernels-l['pad'];valid=(source>=0)&(source<t*4);source=source//4
                else:source=positions*l['stride']+kernels-l['pad'];valid=np.ones(source.shape,bool)
                valid=valid & (source>=0) & (source<t)
                self.indices[index]=(mx.array(source.clip(0,t-1).astype(np.int32)),mx.array(valid[:,:,None]))
            if self.china:t=rows
            elif index<7:t//=2
            elif l['kind']==2:t*=2
        mx.eval([(l['weight'],l.get('packed',mx.array(0)),l['bias']) for l in self.layers])
        self.forward=mx.compile(self._forward) if compile_graph else self._forward
        self.batch_tile=128
        self.forward_batch=mx.compile(self._forward_batch) if compile_graph else self._forward_batch
    def _linear(self,x,index):
        l=self.layers[index]
        if self.bits==8:
            if x.shape[-1]!=l['qwidth']:x=mx.pad(x,((0,0),(0,l['qwidth']-x.shape[-1])))
            return mx.quantized_matmul(x,l['packed'],l['qscales'],l['qbias'],group_size=64,bits=8)+l['bias']
        if x.shape[-1]!=l['width']:x=mx.pad(x,((0,0),(0,l['width']-x.shape[-1])))
        return x@l['weight'].T+l['bias']
    def _conv(self,x,index,relu=False):
        if index in self.indices:
            indices,mask=self.indices[index];x=mx.where(mask,x[indices],0).reshape(indices.shape[0],-1)
        y=self._linear(x,index)
        return mx.maximum(y,0) if relu else y
    def _forward(self,x):
        if self.china:return self._forward_batch(x[None])[0]
        for index in range(7):
            x=self._conv(x,index,True);x=mx.max(x.reshape(-1,2,x.shape[-1]),axis=1)
        for base in (7,12):
            directions=[]
            for direction in range(2):
                ih=self._conv(x,base+direction*2);l=self.layers[base+direction*2+1]
                directions.append(self.recur(inputs=[ih,l['weight'],l['qweight'],l['scale'],l['bias']],
                    template=[('STEPS',x.shape[0]),('REVERSE',bool(direction)),('QUANT',self.bits==8)],
                    grid=(384,1,1),threadgroup=(384,1,1),output_shapes=[(x.shape[0],96)],output_dtypes=[mx.float32])[0])
            x=self._conv(mx.concatenate(directions,axis=1),base+4)
        for index in range(17,25):x=self._conv(x,index,index!=24)
        return mx.softmax(x,axis=1)
    def _predict_window(self,x,kind,start,diff):
        normalized=np.empty((self.options.window,3),np.float32)
        status=self.lib.pnsn_preprocess(x.ctypes.data,kind,len(x),start,self.options.window,diff,normalized.ctypes.data)
        if status:raise RuntimeError(self.lib.pnsn_last_error().decode())
        out=self.forward(mx.array(normalized));mx.eval(out);result=np.array(out)
        if not np.isfinite(result).all():raise RuntimeError("MLX produced nonfinite probabilities")
        return result
    def predict(self,x):
        x,kind=waveform(x)
        if len(x)!=self.options.window:raise ValueError('Fixed inference requires samples == configured window')
        return self._predict_window(x,kind,0,0)
    def pick(self,x):
        x,kind=waveform(x)
        if not len(x):return np.empty((0,3))
        candidates=[];used=0;start=0;w=self.options.window;step=w-self.options.overlap
        while True:
            for diff in range(self.options.difference+1):
                p=self._predict_window(x,kind,start,diff)[:min(w,len(x)-start),1:]
                times,phases=np.where(p>self.options.threshold);used+=len(times)
                if used>self.options.max_candidates:raise RuntimeError('Candidate memory budget exceeded')
                candidates.extend(zip(phases,times+start,p[times,phases]))
            if len(x)-start<=w:break
            start+=step
        c=(Pick*used)(*(Pick(int(p),int(t),float(s)) for p,t,s in candidates));out=(Pick*used)();n=ct.c_size_t()
        status=self.lib.pnsn_nms(c,used,self.options.min_gap,out,used,ct.byref(n))
        if status:raise RuntimeError(self.lib.pnsn_last_error().decode())
        return np.array([(out[i].phase,out[i].sample,out[i].confidence) for i in range(n.value)]).reshape(-1,3)

    def _batch_conv_tile(self,x,index,relu=False):
        batch=x.shape[0];rows=x.shape[1]
        if index in self.indices:
            indices,mask=self.indices[index];rows=indices.shape[0]
            x=mx.where(mask[None],x[:,indices,:],0).reshape(batch*rows,-1)
        else:x=x.reshape(batch*rows,-1)
        out=self._linear(x,index).reshape(batch,rows,-1)
        return mx.maximum(out,0) if relu else out
    def _batch_conv(self,x,index,relu=False):
        if index not in self.indices:return self._batch_conv_tile(x,index,relu)
        # Per-layer im2col tiling bounds memory; all records retain a single batch graph.
        return mx.concatenate([self._batch_conv_tile(x[start:start+self.batch_tile],index,relu)
                               for start in range(0,x.shape[0],self.batch_tile)],axis=0)
    def _forward_batch(self,x):
        if self.china:
            skips=[]
            for i in range(6):
                x=self._batch_conv(x,i,True)
                if 1<=i<=4:skips.append(x)
            for i,skip in zip(range(6,10),skips[::-1]):x=mx.concatenate([skip,self._batch_conv(x,i,True)],axis=2)
            return mx.softmax(self._batch_conv(x,10),axis=2)

        batch=x.shape[0]
        for index in range(7):
            x=self._batch_conv(x,index,True)
            x=mx.max(x.reshape(batch,-1,2,x.shape[-1]),axis=2)
        for base in (7,12):
            directions=[];steps=x.shape[1]
            for direction in range(2):
                ih=self._batch_conv(x,base+direction*2);l=self.layers[base+direction*2+1]
                directions.append(self.recur(inputs=[ih,l['weight'],l['qweight'],l['scale'],l['bias']],
                    template=[('STEPS',steps),('REVERSE',bool(direction)),('QUANT',self.bits==8)],
                    grid=(batch*384,1,1),threadgroup=(384,1,1),output_shapes=[(batch,steps,96)],output_dtypes=[mx.float32])[0])
            x=self._batch_conv(mx.concatenate(directions,axis=2),base+4)
        for index in range(17,25):x=self._batch_conv(x,index,index!=24)
        return mx.softmax(x,axis=2)
    def _batch_probabilities(self,x,kind,diff=0,normalized=False):
        if normalized:
            if kind!=0:raise ValueError('Normalized input must be float32')
            data=mx.transpose(mx.array(x),(0,2,1))
        else:
            data=np.empty((len(x),self.options.window,3),np.float32)
            status=self.lib.pnsn_preprocess_batch(x.ctypes.data,kind,len(x),self.options.window,diff,data.ctypes.data)
            if status:raise RuntimeError(self.lib.pnsn_last_error().decode())
            data=mx.array(data)
        y=self.forward_batch(data);mx.eval(y);result=np.array(y).transpose(0,2,1)
        if not np.isfinite(result).all():raise RuntimeError('MLX produced nonfinite batch probabilities')
        return result
    def predict_batch(self,x,normalized=False):
        x,kind=raw_array(x)
        if x.ndim!=3 or x.shape[1:]!=(3,self.options.window):raise ValueError('Realtime input must be [B,3,configured_window]')
        if not len(x):return np.empty((0,self.classes,self.options.window),np.float32)
        return self._batch_probabilities(x,kind,normalized=normalized)
    def pick_history(self,x):
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[0]!=3:raise ValueError('Historical input must be [3,N]')
        return self.pick(x.T)
    def pick_batch(self,x):
        x,kind=raw_array(x)
        if x.ndim!=3 or x.shape[1:]!=(3,self.options.window):raise ValueError('Realtime input must be [B,3,configured_window]')
        if not len(x):return []
        probs=[self._batch_probabilities(x,kind)]
        if self.options.difference:probs.append(self._batch_probabilities(x,kind,1))
        result=[]
        for b in range(len(x)):
            candidates=[]
            for prob in probs:
                p=prob[b,1:,:].T;times,phases=np.where(p>self.options.threshold)
                candidates.extend(zip(phases,times,p[times,phases]))
            n=len(candidates)
            if n>self.options.max_candidates:raise RuntimeError('Per-record candidate memory budget exceeded')
            c=(Pick*n)(*(Pick(int(p),int(t),float(s)) for p,t,s in candidates));out=(Pick*n)();count=ct.c_size_t()
            status=self.lib.pnsn_nms(c,n,self.options.min_gap,out,n,ct.byref(count))
            if status:raise RuntimeError(self.lib.pnsn_last_error().decode())
            result.append(np.array([(out[i].phase,out[i].sample,out[i].confidence) for i in range(count.value)],np.float64).reshape(-1,3))
        return result
