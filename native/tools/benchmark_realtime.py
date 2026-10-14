# SPDX-License-Identifier: GPL-3.0-only
"""Complete near-real-time picking, CPU context pool or cuDNN, real public waveform."""
import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pnsn import Picker

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend', choices=['cpu', 'cudnn', 'cudnn_int8'], required=True)
p.add_argument('--models', nargs='+', choices=['pnsn', 'china-unet'], default=['pnsn', 'china-unet'])
p.add_argument('--precision', choices=['fp32', 'int8'], default='int8')
p.add_argument('--workers', nargs='+', type=int, default=[1, 8, 16])
p.add_argument('--batch', type=int, default=4096)
p.add_argument('--length', type=int, default=6144)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--input', default=str(ROOT / 'reports/real-waveform.npy'))
p.add_argument('--output', required=True)
a = p.parse_args()
if a.batch < 1 or a.repeats < 1 or any(w < 1 or w > a.batch for w in a.workers):
    p.error('Batch/repeats/workers must be positive; workers <= batch')
wave = np.load(a.input, allow_pickle=False)
if wave.ndim != 2 or wave.shape[1] != 3 or len(wave) < a.length or not np.isfinite(wave).all():
    raise ValueError('Expected finite interleaved [N,3] real E/N/Z waveform, N >= length')
rng = np.random.default_rng(20261014)
starts = rng.integers(0, len(wave) - a.length + 1, size=a.batch)
x = np.empty((a.batch, 3, a.length), np.float32)
for i, start in enumerate(starts):
    record = wave[start:start + a.length].T.astype(np.float64)
    noise_scale = record.std(axis=1, keepdims=True) * .005
    x[i] = record + rng.standard_normal(record.shape) * noise_scale
rows = []
for model in a.models:
    for workers in (a.workers if a.backend == 'cpu' else [1]):
        library = None if a.backend == 'cpu' else ROOT / f'build/libpnsn_{a.backend}.so'
        contexts = [Picker(ROOT / f'models/{model}-{a.precision}.bin', library,
                           window=a.length, threads=1) for _ in range(workers)]
        boundaries = np.linspace(0, a.batch, workers + 1, dtype=int)
        pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

        def one_chunk(worker):
            return contexts[worker].pick_batch(x[boundaries[worker]:boundaries[worker + 1]])

        def run():
            if pool is None:
                return one_chunk(0)
            futures = [pool.submit(one_chunk, worker) for worker in range(workers)]
            return [picks for future in futures for picks in future.result()]

        # Prewarm every context on its own input; GPU additionally reserves the full batch.
        for worker, context in enumerate(contexts):
            context.pick_batch(x[boundaries[worker]:boundaries[worker] + 1])
        if a.backend != 'cpu':
            run()
        times, counts = [], []
        for repeat in range(a.repeats):
            begin = time.perf_counter()
            out = run()
            elapsed = time.perf_counter() - begin
            assert len(out) == a.batch
            assert all(np.isfinite(v).all() and ((v[:, 1] >= 0) & (v[:, 1] < a.length)).all() for v in out)
            count = sum(len(v) for v in out)
            times.append(elapsed)
            counts.append(count)
            print(json.dumps(dict(model=model, backend=a.backend, precision=a.precision,
                                  workers=workers, repeat=repeat, seconds=elapsed, picks=count)), flush=True)
        independent_picks_equal = []
        for index in [0, a.batch // 2, a.batch - 1]:
            independent = contexts[0].pick_batch(x[index:index + 1])[0]
            equal = bool(np.array_equal(out[index][:, :2], independent[:, :2]))
            independent_picks_equal.append(equal)
            if a.backend == 'cpu':
                assert equal
        assert len(set(counts)) == 1, counts
        row = dict(model=model, backend=a.backend, precision=a.precision,
                   compute='FP32' if a.precision == 'fp32' else 'Q8 storage / FP32 compute' if a.backend == 'cudnn'
                   else 'conv W8A8 / LSTM FP32' if a.backend == 'cudnn_int8' and model == 'pnsn'
                   else 'W8A8 GEMM / float activations', workers=workers,
                   threads_per_context=1 if a.backend == 'cpu' else None,
                   batch=a.batch, length=a.length, seconds=times, picks=counts[0],
                   median_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times, 95)),
                   windows_per_second=a.batch / float(np.median(times)),
                   independent_picks_equal=independent_picks_equal)
        if a.backend != 'cpu':
            # The independent validation call changed B; restore full-B allocation/statistics.
            run()
            row.update(contexts[0].batch_device_stats())
        rows.append(row)
        if pool:
            pool.shutdown(wait=True)
        for context in contexts:
            context.close()
        del contexts, out
        gc.collect()

result = dict(host=platform.node(), platform=platform.platform(), seed=20261014,
              input=f'{a.batch} independently selected windows from upstream public SAC E/N/Z example plus 0.5% channel-std noise; raw float32 [B,3,L]',
              input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
              batch=a.batch, length=a.length, reference_sampling_rate_hz=100,
              scope='Complete raw host input to independent picks, includes internal preprocessing, model inference, min-heap NMS, GPU transfers/synchronization and CPU pool submission; excludes loading/data generation/validation; persistent contexts and pool; each CPU worker has its own context with threads=1',
              records=rows)
Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({**result, 'records': rows}, indent=2))
