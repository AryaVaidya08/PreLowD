---
name: PreLowD Research Project
description: Core goal and architecture decisions for PreLowD — FFNO-based transfer learning for PDE simulations
type: project
---

Core research goal: proof-of-concept hypernetwork system for 1D→2D→3D transfer learning in physics simulations (fluid dynamics / CFD). Currently evaluating an FFNO baseline.

Key architectural decisions:
- Models.py: original FFNO (no PDE params)
- Models_PDE.py: FFNO_PDE with FiLM conditioning — PDEConditioner is a 2-layer MLP (param_dim→width→ReLU→2*width) producing per-channel γ/β; applied in each Factorized_Spectral_Layer_PDE after spectral sum, before feedforward. Zero-init on output layer for identity start.
- Train_FFNO_PDE.py + configs_FFNO/_base_pde.yaml: training pipeline for FFNO_PDE
- PDE params loaded per-trajectory from h5 file (pde_param_vars) or as constants (pde_param_values)
- ParametrizedPDEDataset wraps PDEDataset; DataLoader yields (xs, pde_params) tuples
- Conditioner excluded from transfer by default (transfer_conditioner_layers='') so 1D→2D transfer works even if param spaces differ

**Why:** vanilla FFNO generalizes poorly because it sees no PDE parameters (e.g. viscosity, Reynolds number), so it blends across parameter regimes. FiLM adds lightweight conditioning without changing in/out interface.

**How to apply:** when adding features or debugging, check Models_PDE.py and Train_FFNO_PDE.py first. The original Models.py / Train_FFNO.py are preserved as-is.
