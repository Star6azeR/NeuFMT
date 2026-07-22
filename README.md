# µNeuFMT

Official implementation for paper: µNeuFMT: Background-Optical-Property-Adaptive Fluorescence Molecular Tomography via Implicit Neural Representation

[![Static Badge](https://img.shields.io/badge/arXiv-2511.04510-red?logo=arXiv)](https://arxiv.org/abs/2511.04510)
![Static Badge](https://shields.io/github/license/apache/geaflow?logo=apache&label=License&color=blue)

<div align="center">
<img src="https://raw.githubusercontent.com/Star6azeR/NeuFMT/main/Concept.png" alt="Concept.png" width="600" /> 
</div>

## NeuFLIM: INR + time-domain lifetime imaging

`NeuFLIM_main.py` extends the non-adaptive NeuFMT path to reconstruct two spatial fields from emission TPSFs:

- fluorescence yield
- fluorescence lifetime

The INR uses one shared coordinate-MLP trunk and two physically bounded output heads. The training loop evaluates a differentiable time-domain forward recurrence at every iteration:

```text
lifetime -> exponential excitation-history state R(t)
yield * R(t) -> fluorescence source
fluorescence source -> emission field -> detector TPSF
```

The excitation field and the emission propagation/source operators are precomputed, so the trainable forward pass only carries the lifetime state, emission state, and detector measurements through time.

### Run

```bash
python NeuFLIM_main.py --config configs/NeuFLIM_peanut_example.txt
```

See `data/NeuFLIM_format.txt` for the required `.mat` files, tensor shapes, operator definitions, and time-unit requirements. NeuFMT-Ada is unchanged.
