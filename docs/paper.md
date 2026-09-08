# Paper

This repository provides code and selected results for:

**Large AI Model and ISAC-Enhanced UAV Network Optimization Through DRL**

IEEE Transactions on Vehicular Technology, DOI: `10.1109/TVT.2026.3705101`.

## Summary

The project studies UAV-assisted wireless communications where ground-terminal locations are imperfectly perceived. It combines VLM-based visual distance estimation and ISAC-based RF sensing with a lightweight gate-fusion module, then uses DDPG to jointly optimize UAV trajectory and communication resource allocation under a computation-aware energy-efficiency objective.

## Main components

- VLM-assisted distance perception for UAV-captured aerial imagery.
- ISAC-based RF distance sensing.
- Gate-based fusion of visual and RF distance estimates.
- DDPG control for UAV 3D trajectory and resource allocation.
- Computation-aware energy model including propulsion, communication, VLM inference, ISAC processing, fusion, and policy execution.

## Reported scenarios

- Nominal urban scenarios: G1--G4, 6 ground terminals, 200 m x 200 m area, 200 m maximum altitude.
- Complex large-scale scenarios: C1--C4, 8 or 12 ground terminals, 1000 m x 1000 m area, 250 m maximum altitude.
