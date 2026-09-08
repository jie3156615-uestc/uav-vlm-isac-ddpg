# VLM-ISAC-DDPG UAV Network Optimization

Code and selected results for the paper:

**Large AI Model and ISAC-Enhanced UAV Network Optimization Through DRL**

IEEE Transactions on Vehicular Technology, DOI: `10.1109/TVT.2026.3705101`.

## Overview

This repository implements a computation-aware UAV communication optimization framework. The method fuses VLM-based visual distance estimation with ISAC-based RF sensing through a lightweight gate module, then uses DDPG to optimize UAV trajectory and communication resource allocation.

Compared methods:

- `DDPG-VLM-ISAC`: proposed multimodal fusion method.
- `DDPG-ISAC`: RF sensing only.
- `DDPG-VLM`: visual sensing only.
- `DDPG-noISAC`: baseline without additional VLM-ISAC perception enhancement.

## Repository structure

```text
.
├── src/                    # DDPG agent and UAV environment modules
├── scripts/                # Training, evaluation, and paper scenario runners
├── results/
│   ├── tables/             # Selected paper result tables
│   └── figures/            # Selected paper figures
├── docs/                   # Paper summary and notes
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch can be installed with either CPU or CUDA support. Follow the official PyTorch installer if you need a specific CUDA build.

## Quick smoke run

The command below runs a small CPU-friendly training job. It is intended to check that the environment and scripts work, not to reproduce the final paper numbers.

```bash
python scripts/DDPG_batch_train.py \
  --profiles nominal \
  --num-ground-stations-list 6 \
  --scene-list urban \
  --train-modes our,isac,vlm,noisac \
  --episodes 2 \
  --steps 5 \
  --output-tag smoke
```

## Reproducing paper-style experiments

Train the main methods:

```bash
python scripts/DDPG_batch_train.py \
  --profiles nominal \
  --num-ground-stations-list 6 \
  --scene-list urban \
  --train-modes our,isac,vlm,noisac \
  --episodes 300 \
  --steps 100 \
  --output-tag nominal
```

Evaluate trained policies:

```bash
python scripts/DDPG_batch_experiments.py \
  --profiles nominal \
  --num-ground-stations-list 6 \
  --scene-list urban \
  --episodes 300 \
  --steps 100 \
  --model-registry batch_train_results/<run>/model_registry.json \
  --output-tag nominal_eval \
  --save-trajectories
```

Large-scale scenario runners are provided in:

- `scripts/run_complex_scenarios.py`
- `scripts/run_paper_extra_groups.py`
- `scripts/run_paper_g4_wide.py`

These scripts may require long training time and generate model checkpoints under `models/`.

## Selected results

The final summarized numbers are stored in:

- `results/tables/final_results.csv`

Key reported observations:

- In nominal urban 6-GT scenarios, `DDPG-VLM-ISAC` reaches rewards around 740--759 and energy efficiency above 41.9 Kbps/J.
- In complex C1, `DDPG-VLM-ISAC` improves reward by 93.2% over `DDPG-ISAC` and 115.5% over `DDPG-VLM`.
- In complex C2, `DDPG-VLM-ISAC` improves reward by 196.0% over `DDPG-noISAC`.

## Notes on checkpoints and datasets

Large generated artifacts such as `.pth` model checkpoints, temporary logs, and full experiment folders are intentionally excluded from Git by `.gitignore`. Regenerate them with the scripts above, or publish them separately through GitHub Releases or Git LFS if needed.

The VLM distance-estimation component is represented in the simulation pipeline through the perception model and fusion mechanism used by the UAV environments. If you release external VLM checkpoints or raw image-text datasets later, add their download links here.

## Citation

If this repository helps your research, please cite:

```bibtex
@article{gao2026vlmisacddpg,
  title={Large AI Model and ISAC-Enhanced UAV Network Optimization Through DRL},
  journal={IEEE Transactions on Vehicular Technology},
  year={2026},
  doi={10.1109/TVT.2026.3705101}
}
```

## License

This project is released under the MIT License.
