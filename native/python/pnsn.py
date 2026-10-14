"""NumPy/ctypes convenience wrapper. C deployments need neither Python nor NumPy."""
import ctypes as ct
import os
import sys
from pathlib import Path
import numpy as np

class Options(ct.Structure):
    _fields_=[('window',ct.c_size_t),('overlap',ct.c_size_t),('max_candidates',ct.c_size_t),('threshold',ct.c_float),('min_gap',ct.c_int64),('threads',ct.c_int),('difference',ct.c_int)]
class Pick(ct.Structure):
    _fields_=[('phase',ct.c_int32),('sample',ct.c_int64),('confidence',ct.c_float)]

def load_library(path=None):
    suffix='dll' if sys.platform=='win32' else 'dylib' if sys.platform=='darwin' else 'so'
    path=path or os.environ.get('PNSN_LIBRARY') or str(Path(__file__).resolve().parents[1]/'build'/('pnsn.dll' if suffix=='dll' else f'libpnsn.{suffix}'))
    lib=ct.CDLL(str(path))
    lib.pnsn_default_options.restype=Options
    lib.pnsn_open.argtypes=[ct.c_char_p,ct.POINTER(Options)];lib.pnsn_open.restype=ct.c_void_p
    lib.pnsn_close.argtypes=[ct.c_void_p]
    lib.pnsn_last_error.restype=ct.c_char_p
    lib.pnsn_preprocess.argtypes=[ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_size_t,ct.c_size_t,ct.c_int,ct.c_void_p]
    lib.pnsn_predict.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_void_p]
    lib.pnsn_preprocess_channels.argtypes=lib.pnsn_preprocess.argtypes
    lib.pnsn_preprocess_batch.argtypes=[ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_size_t,ct.c_int,ct.c_void_p]
    lib.pnsn_predict_batch.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_size_t,ct.c_void_p]
    lib.pnsn_predict_batch_normalized.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_size_t,ct.c_void_p]
    lib.pnsn_pick_batch.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_size_t,ct.c_size_t,ct.POINTER(Pick),ct.c_size_t,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)]
    lib.pnsn_batch_device_stats.argtypes=[ct.c_void_p,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.POINTER(ct.c_size_t)]
    lib.pnsn_pick_trace.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_size_t,ct.POINTER(Pick),ct.c_size_t,ct.POINTER(ct.c_size_t)]
    lib.pnsn_pick_history.argtypes=lib.pnsn_pick_trace.argtypes
    lib.pnsn_nms.argtypes=[ct.POINTER(Pick),ct.c_size_t,ct.c_int64,ct.POINTER(Pick),ct.c_size_t,ct.POINTER(ct.c_size_t)]
    lib.pnsn_cpu_features.restype=ct.c_char_p
    lib.pnsn_classes.argtypes=[ct.c_void_p];lib.pnsn_classes.restype=ct.c_int
    lib.pnsn_model_name.argtypes=[ct.c_void_p];lib.pnsn_model_name.restype=ct.c_char_p
    lib.pnsn_memory_bytes.argtypes=[ct.c_void_p];lib.pnsn_memory_bytes.restype=ct.c_size_t
    return lib

def raw_array(x):
    x=np.asarray(x)
    if x.dtype==np.int32:kind=2
    elif x.dtype==np.float32:kind=0
    else:x=x.astype(np.float64);kind=1
    if not x.dtype.isnative:x=x.astype(x.dtype.newbyteorder('='))
    return np.ascontiguousarray(x),kind

def waveform(x):
    x,kind=raw_array(x)
    if x.ndim!=2 or x.shape[1]!=3:raise ValueError('Expected interleaved [samples,3] E/N/Z')
    return x,kind


class Picker:
    def __init__(self,model,library=None,**options):
        self.lib=load_library(library);self.options=self.lib.pnsn_default_options()
        for k,v in options.items():
            if k not in {n for n,_ in Options._fields_}:raise ValueError(f'Unknown option {k}')
            setattr(self.options,k,v)
        self.ctx=self.lib.pnsn_open(os.fsencode(model),ct.byref(self.options))
        if not self.ctx:raise RuntimeError(self.lib.pnsn_last_error().decode())
        self.classes=self.lib.pnsn_classes(self.ctx);self.model_name=self.lib.pnsn_model_name(self.ctx).decode()
    def close(self):
        if getattr(self,'ctx',None):self.lib.pnsn_close(self.ctx);self.ctx=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):self.close()
    def _check(self,status):
        if status:raise RuntimeError(self.lib.pnsn_last_error().decode())
    def predict(self,x):
        if not self.ctx:raise RuntimeError('Picker is closed')
        x,kind=waveform(x)
        if len(x)!=self.options.window:raise RuntimeError("Fixed inference requires samples == configured window")
        y=np.empty((len(x),self.classes),np.float32)
        self._check(self.lib.pnsn_predict(self.ctx,x.ctypes.data,kind,len(x),y.ctypes.data));return y
    def pick(self,x,capacity=None):
        if not self.ctx:raise RuntimeError('Picker is closed')
        x,kind=waveform(x)
        capacity=capacity or min(self.options.max_candidates,(self.classes-1)*(len(x)//(self.options.min_gap+1)+1))
        out=(Pick*capacity)();count=ct.c_size_t()
        self._check(self.lib.pnsn_pick_trace(self.ctx,x.ctypes.data,kind,len(x),out,capacity,ct.byref(count)))
        return np.array([(out[i].phase,out[i].sample,out[i].confidence) for i in range(count.value)],np.float64).reshape(-1,3)

    def pick_history(self,x,capacity=None):
        if not self.ctx:raise RuntimeError('Picker is closed')
        x,kind=raw_array(x)
        if x.ndim!=2 or x.shape[0]!=3:raise ValueError('Historical input must be [3,N] E/N/Z')
        length=x.shape[1];capacity=capacity or min(self.options.max_candidates,(self.classes-1)*(length//(self.options.min_gap+1)+1))
        out=(Pick*capacity)();count=ct.c_size_t()
        self._check(self.lib.pnsn_pick_history(self.ctx,x.ctypes.data,kind,length,out,capacity,ct.byref(count)))
        return np.array([(out[i].phase,out[i].sample,out[i].confidence) for i in range(count.value)],np.float64).reshape(-1,3)
    def predict_batch(self,x,normalized=False):
        if not self.ctx:raise RuntimeError('Picker is closed')
        x,kind=raw_array(x)
        if x.ndim!=3 or x.shape[1:]!=(3,self.options.window):raise ValueError('Realtime input must be [B,3,configured_window]')
        if normalized and kind!=0:raise ValueError('Normalized expert input must be float32')
        y=np.empty((len(x),self.options.window,self.classes),np.float32)
        if normalized:self._check(self.lib.pnsn_predict_batch_normalized(self.ctx,x.ctypes.data,len(x),self.options.window,y.ctypes.data))
        else:self._check(self.lib.pnsn_predict_batch(self.ctx,x.ctypes.data,kind,len(x),self.options.window,y.ctypes.data))
        return y.transpose(0,2,1)
    def batch_device_stats(self):
        p=ct.c_double();n=ct.c_double();b=ct.c_size_t()
        self._check(self.lib.pnsn_batch_device_stats(self.ctx,ct.byref(p),ct.byref(n),ct.byref(b)))
        return dict(preprocess_ms=p.value,network_ms=n.value,device_buffer_bytes=b.value)
    def pick_batch(self,x,capacity=None):
        if not self.ctx:raise RuntimeError('Picker is closed')
        x,kind=raw_array(x)
        if x.ndim!=3 or x.shape[1:]!=(3,self.options.window):raise ValueError('Realtime input must be [B,3,configured_window]')
        per=min(self.options.max_candidates,(self.classes-1)*(self.options.window//(self.options.min_gap+1)+1));capacity=capacity or len(x)*per
        out=(Pick*capacity)();bounds=(ct.c_size_t*(len(x)+1))();count=ct.c_size_t()
        self._check(self.lib.pnsn_pick_batch(self.ctx,x.ctypes.data,kind,len(x),self.options.window,out,capacity,bounds,ct.byref(count)))
        flat=np.array([(out[i].phase,out[i].sample,out[i].confidence) for i in range(count.value)],np.float64).reshape(-1,3)
        return [flat[bounds[i]:bounds[i+1]] for i in range(len(x))]
