# SPDX-License-Identifier: GPL-3.0-only
"""Measure the complete arbitrary-length history path on a real E/N/Z trace."""
import argparse, gc, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pnsn import Picker

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['cpu', 'cudnn', 'cudnn_int8', 'mlx'], required=True)
p.add_argument('--input', default=str(ROOT / 'reports/real-waveform.npy'))
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--output', required=True)
a = p.parse_args()
if a.repeats < 1:
    p.error('--repeats must be positive')
raw = np.load(a.input, allow_pickle=False)
if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
    raise ValueError('Expected finite [N,3] E/N/Z waveform')
x = np.ascontiguousarray(raw.T, dtype=np.float64)
rows = []
for model in ['pnsn', 'china-unet']:
    for mode in ['fp32', 'int8']:
        if a.backend == 'mlx':
            import mlx.core as mx
            from pnsn_mlx import MLXPicker
            q = MLXPicker(ROOT / f'models/{model}-{mode}.bin')
        else:
            library = None if a.backend == 'cpu' else ROOT / f'build/libpnsn_{a.backend}.so'
            q = Picker(ROOT / f'models/{model}-{mode}.bin', library, threads=1)
        for _ in range(2):
            q.pick_history(x)
        times = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            out = q.pick_history(x)
            times.append((time.perf_counter() - start) * 1000)
        assert np.isfinite(out).all()
        assert ((out[:, 1] >= 0) & (out[:, 1] < x.shape[1])).all()
        compute = ('FP32' if mode == 'fp32' else 'W8A32' if a.backend == 'mlx'
                   else 'Q8 storage / FP32 compute' if a.backend == 'cudnn'
                   else 'conv W8A8 / LSTM FP32' if a.backend == 'cudnn_int8' and model == 'pnsn'
                   else 'W8A8 GEMM / float activations')
        rows.append(dict(model=model, precision=mode, compute=compute,
                         samples=x.shape[1], window=10240, overlap=1024,
                         threads=1 if a.backend == 'cpu' else None, picks=len(out),
                         median_ms=float(np.median(times)),
                         p95_ms=float(np.percentile(times, 95)), times_ms=times))
        if hasattr(q, 'close'):
            q.close()
        del q
        gc.collect()
        if a.backend == 'mlx':
            mx.clear_cache()
result = dict(backend=a.backend, host=platform.node(), platform=platform.platform(),
              input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
              input='upstream public three-SAC E/N/Z example, 100 Hz, [3,N] float64',
              scope='complete history call: sliding windows, internal preprocessing, raw probabilities, global min-heap NMS; GPU synchronized; 2 warmups then measured repeats',
              records=rows)
Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({**result, 'records': [{k: v for k, v in row.items() if k != 'times_ms'} for row in rows]}, indent=2))
