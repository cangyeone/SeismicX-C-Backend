"""Convert the upstream SAC example into raw E/N/Z inputs for native benchmarks."""
import numpy as np
from pathlib import Path
import obspy
ROOT=Path(__file__).resolve().parents[2]
traces={}
for path in (ROOT/'data/waveform').glob('*53085*.sac'):
    tr=obspy.read(str(path))[0];traces[tr.stats.channel[-1]]=tr
if set(traces)!=set('ENZ'):raise ValueError('Expected exactly one E/N/Z sample set')
if len({(t.stats.starttime.timestamp,t.stats.sampling_rate,t.stats.npts) for t in traces.values()})!=1:raise ValueError('Component timing/length mismatch')
if any(t.stats.sampling_rate!=100 for t in traces.values()):raise ValueError('Example requires 100 Hz data')
x=np.stack([traces[c].data for c in 'ENZ'],1).astype(np.float64)
out=ROOT/'native/reports';out.mkdir(parents=True,exist_ok=True)
np.save(out/'real-waveform.npy',x);x.tofile(out/'wave-f64.raw');x[:10240].tofile(out/'window-f64.raw')
print(f'Prepared {len(x)} samples, E/N/Z, 100 Hz')
