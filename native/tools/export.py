"""Fuse all 14 BN layers and export the exact BRNNPNSN v3 graph (no runtime torch)."""
import argparse, hashlib, json, struct, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.BRNNPNSN import BRNN
MAGIC = b'PNSNC01\0'

def load_network(checkpoint):
    model = BRNN().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model

def layers(model):
    result = []
    def conv(c, bn=None):
        trans = isinstance(c, torch.nn.ConvTranspose1d)
        w = c.weight.detach().double()
        b = c.bias.detach().double()
        if bn is not None:
            scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
            w = w * scale.reshape((1, -1, 1) if trans else (-1, 1, 1))
            b = (b - bn.running_mean.double()) * scale + bn.bias.detach().double()
        w = w.permute(1, 2, 0) if trans else w.permute(0, 2, 1)
        result.append((2 if trans else 1, c.in_channels, c.out_channels,
                       c.kernel_size[0], c.stride[0], c.padding[0],
                       w.reshape(c.out_channels, -1).float().numpy(), b.float().numpy()))
    for block in model.encoder.layers:
        if hasattr(block, 'layers'):
            conv(block.layers[0], block.layers[1])
    for block in model.rnns:
        for suffix in ['', '_reverse']:
            for kind in ['ih', 'hh']:
                w = getattr(block.rnn, f'weight_{kind}_l0{suffix}').detach().numpy()
                b = (getattr(block.rnn, f'bias_ih_l0{suffix}') + getattr(block.rnn, f'bias_hh_l0{suffix}')).detach().numpy() if kind == 'ih' else np.zeros(384, np.float32)
                result.append((0, w.shape[1], w.shape[0], 1, 1, 0, w, b))
        conv(block.cnn)
    for block in model.decoder.layers:
        if hasattr(block, 'layers'):
            conv(block.layers[0], block.layers[1])
        else:
            conv(block)
    return result

def export(checkpoint, output, precision):
    model = load_network(checkpoint)
    records = layers(model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as f:
        f.write(MAGIC + struct.pack('<II', len(records), 8 if precision == 'int8' else 32))
        for kind, ni, no, k, stride, pad, w, b in records:
            width = ((ni * k + 15) // 16) * 16
            padded = np.zeros((no, width), np.float32)
            padded[:, :ni*k] = w
            scale = np.maximum(np.abs(padded).max(axis=1) / 127, np.finfo(np.float32).tiny)
            data = np.clip(np.rint(padded / scale[:, None]), -127, 127).astype(np.int8) if precision == 'int8' else padded.astype('<f4')
            f.write(struct.pack('<7I', kind, ni, no, k, stride, pad, width))
            f.write(scale.astype('<f4').tobytes())
            f.write(b.astype('<f4').tobytes())
            f.write(data.tobytes())
    info = dict(format=1, precision=precision, fused_bn=14, layers=len(records),
                checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size,
                phases=['Noise', 'Pg', 'Sg', 'Pn', 'Sn'], quantization='symmetric per-output-channel weights; dynamic per-row activations on C/CUDA; float state')
    output.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(ROOT / 'ckpt/pnsn.v3.pt'))
    p.add_argument('--output', required=True)
    p.add_argument('--precision', choices=['fp32', 'int8'], default='int8')
    a = p.parse_args()
    export(a.checkpoint, a.output, a.precision)
