# Model and implementation provenance

Reference architectures and checkpoints are from the same GitHub owner's
[cangyeone/pnsn](https://github.com/cangyeone/pnsn), source commit
`1f955f4d8cffbc5c43584f02ef14a0da2b0aeb6b`.

- PNSN: `models/BRNNPNSN.py`, checkpoint `ckpt/pnsn.v3.pt`, SHA256 `9f626e5fff4e9390c88e43c2f6230802496163b5d6eefee05e1b6ac7ccebf9e8`.
- China PhaseNet-style UNet: `models/UNet.py`, checkpoint `ckpt/china.unet.pt`, SHA256 `ec74a633b2c9a448089b05ab3a4ea54ef7377c2b961d1518897c89bc3e7d9b00`.

The China UNet is the selected nearest-neighbor upsampling architecture, not
the original PhaseNet implementation. PhaseNet's original method is described
in Zhu and Beroza (2019), *PhaseNet: a deep-neural-network-based seismic
arrival-time picking method*, [doi:10.1093/gji/ggy423](https://doi.org/10.1093/gji/ggy423).

The native CPU inference, versioned containers, C preprocessing, heap NMS,
cuDNN integration, MLX adapter and tests in this repository were implemented
for SeismicX. No SeisBench source code is included or required. PyTorch is used
only for model export and reference testing. cuDNN and MLX are optional external
backends and retain their own terms; their binaries are not bundled.

Reference example SAC files are taken from the upstream public repository.
Private training/test caches are not included. Evaluation reports publish
aggregate statistics and reproducible commands; paths and host identifiers
are generalized for publication.

## License

Copyright (C) 2026 SeismicX contributors (cangyeone).

This release is distributed under the GNU General Public License, version 3
only (`SPDX-License-Identifier: GPL-3.0-only`). The project source and included
reference checkpoints/exported model files are covered by this release's GPLv3
license. See [LICENSE](LICENSE) for the complete license text. Optional external
cuDNN, CUDA and MLX dependencies retain their own licenses.
