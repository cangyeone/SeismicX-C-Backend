# SPDX-License-Identifier: GPL-3.0-only
"""Strict reader for the versioned little-endian native model container."""
import struct
from pathlib import Path
import numpy as np

def read_model(path):
    data=Path(path).read_bytes()
    if data[:8] not in (b'PNSNC01\0',b'SXUNET1\0') or len(data)<16:raise ValueError('Invalid PNSN container')
    count,bits=struct.unpack_from('<II',data,8)
    if count!=(11 if data[:8]==b'SXUNET1\0' else 25) or bits not in (8,32):raise ValueError('Incompatible model')
    offset=16;result=[]
    for i in range(count):
        kind,ni,no,k,stride,pad,width=struct.unpack_from('<7I',data,offset);offset+=28
        if not 0<ni<=192 or not 0<no<=384 or not 0<k<=11 or width!=((ni*k+15)//16)*16:raise ValueError('Invalid model dimensions')
        scale=np.frombuffer(data,dtype='<f4',count=no,offset=offset).copy();offset+=no*4
        bias=np.frombuffer(data,dtype='<f4',count=no,offset=offset).copy();offset+=no*4
        w=np.frombuffer(data,dtype=np.int8 if bits==8 else '<f4',count=no*width,offset=offset).copy().reshape(no,width)
        offset+=no*width*(1 if bits==8 else 4)
        if not np.isfinite(scale).all() or not np.isfinite(bias).all() or (scale<=0).any() or not np.isfinite(w).all():raise ValueError('Invalid model values')
        result.append(dict(kind=kind,ni=ni,no=no,k=k,stride=stride,pad=pad,width=width,w=w,bias=bias,scale=scale))
    if offset!=len(data):raise ValueError('Unexpected model trailing data')
    return bits,result
