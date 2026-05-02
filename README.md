# DrugGen-RL

Reinforcement learning framework for *de novo* drug molecule generation. An RL agent learns to build valid, drug-like molecules token by token in SMILES notation, guided by a molecular property scorer as its reward signal.

## How it works

```
Reset episode
     │
     ▼
┌────────────────────────────────────────┐
│  MoleculeEnv  (Gymnasium)              │
│  State: one-hot token history (1920-d) │
│  Action: next SMILES token (vocab=32)  │
│  Reward: scorer(SMILES) at terminal    │
└────────────────────────────────────────┘
     │  action (token index)
     ▼
┌──────────────────────┐
│  RL Agent            │
│  PPO  – RecurrentPPO │  LSTM(128) + MLP(128,128)
│       – MaskablePPO  │  optional action masking
│  SAC  – continuous   │  actor argmax → discrete token
└──────────────────────┘
     │  terminal reward
     ▼
┌──────────────────────┐
│  Molecular Scorer    │
│  QED / SA / LogP /   │
│  MW / Tanimoto /     │
│  Multi-Objective     │
└──────────────────────┘
```

At each step the agent picks one token from a 32-symbol vocabulary (atoms, bonds, ring-digits, brackets). When it emits `<STOP>` or hits the 60-token limit the episode ends, the accumulated SMILES is validated with RDKit, and the scorer returns a reward in `[0, 1]`. Action masking constrains the agent to chemically plausible transitions, preventing double-bonds after bonds, unclosed parentheses, etc.

## Installation

```bash
git clone https://github.com/rajul-kk/Drug_Generation.git
cd Drug_Generation
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, RDKit, Stable-Baselines3 ≥ 2.0, sb3-contrib ≥ 2.0

## Quick start

### Train

```bash
# PPO with LSTM (default, QED objective)
python train_agent.py --agent ppo --scorer qed --timesteps 600000

# PPO with action masking (MaskablePPO)
python train_agent.py --agent ppo --scorer qed --timesteps 600000 --mask-actions

# SAC (continuous action space, QED objective)
python train_agent.py --agent sac --scorer qed --timesteps 300000

# Multi-objective (QED + SA score)
python train_agent.py --agent ppo --scorer multi --timesteps 600000
```

Resume from a checkpoint:

```bash
python train_agent.py --agent ppo --scorer qed --resume checkpoints/ppo_qed/best_model/best_model.zip
```

### Evaluate

```bash
# Evaluate the included PPO-QED checkpoint (100 molecules)
python evaluate_agent.py \
    --agent ppo \
    --model checkpoints/ppo_qed/best_model/best_model.zip \
    --scorer qed \
    --episodes 100 \
    --mask-actions

# With a reference set for novelty scoring
python evaluate_agent.py \
    --agent ppo \
    --model checkpoints/ppo_qed/best_model/best_model.zip \
    --scorer qed \
    --episodes 200 \
    --reference-file data/chembl_reference.smi
```

### Compute GuacaMol-style metrics from a SMILES file

```bash
python metrics_from_smiles.py \
    --input generated.smi \
    --reference-file data/chembl_reference.smi \
    --topk 1,10,100
```

## Scorers

| Name | Description | Range |
|------|-------------|-------|
| `qed` | Quantitative Estimate of Drug-likeness | [0, 1] |
| `sa` | Synthetic Accessibility (inverted RDKit SA score) | [0, 1] |
| `logp` | LogP Gaussian window (target 1–4) | [0, 1] |
| `mw` | Molecular weight Gaussian window (target 250–500 Da) | [0, 1] |
| `tanimoto` | Tanimoto similarity to a target SMILES | [0, 1] |
| `isomer` | Exact structural isomer match (InChI Key) | {0, 1} |
| `rediscovery` | Canonical SMILES exact match | {0, 1} |
| `multi` | Weighted geometric / arithmetic mean of any scorers | [0, 1] |

Adding a custom scorer:

```python
from core.scoring import MolecularScorer, register_scorer

class MyScorer(MolecularScorer):
    def score(self, smiles: str) -> float:
        ...  # return float in [0, 1]

register_scorer("my_scorer", MyScorer)
```

## Results

> **Note:** Run `evaluate_agent.py` with your trained checkpoint and fill in the table below.

| Model | Scorer | Timesteps | Validity | Uniqueness | Novelty | Diversity | Avg Score |
|-------|--------|-----------|----------|------------|---------|-----------|-----------|
| PPO-LSTM | QED | 600k | — | — | — | — | — |
| PPO-Masked | QED | 600k | — | — | — | — | — |
| SAC | QED | 300k | — | — | — | — | — |
| Random baseline | — | — | ~5% | ~100% | ~100% | ~0.85 | ~0.05 |

Metrics follow GuacaMol conventions:
- **Validity** – % of generated SMILES that parse and sanitize in RDKit
- **Uniqueness** – % unique canonical SMILES among valid molecules
- **Novelty** – % of valid unique molecules not in the reference set
- **Diversity** – 1 − mean pairwise Tanimoto (Morgan FP, r=2, 2048 bits)
- **Avg Score** – mean scorer value across all valid molecules

## Project structure

```
DrugGen_RL/
├── agents/
│   ├── PPO.py          # RecurrentPPO (LSTM) and MaskablePPO wrappers
│   └── SAC.py          # Soft Actor-Critic wrapper
├── core/
│   ├── chemistry.py    # RDKit utilities (validate, canonicalize, fingerprint)
│   └── scoring.py      # Scorer ABC + registry + all built-in scorers
├── envs/
│   └── molecule_env.py # Gymnasium environment (token-level SMILES builder)
├── checkpoints/        # Saved model weights (.zip)
├── train_agent.py      # Unified training entry point
├── evaluate_agent.py   # Inference + metrics CLI
├── metrics_from_smiles.py  # GuacaMol-style metrics from a SMILES file
└── requirements.txt
```

## Vocabulary

32 tokens covering common drug-like SMILES:

```
<PAD> <STOP>
C N O S P F Cl Br I          # aliphatic atoms
c n o s p                    # aromatic atoms
1 2 3 4 5 6                  # ring-closure digits
( ) [ ] = # - +              # structure / bonds
[nH] [O-] [N+] [nH+]        # common bracket atoms
```

## Training tips

- **Action masking** (`--mask-actions`) significantly improves validity early in training by preventing illegal token transitions.
- **Duplicate penalty** (`--duplicate-penalty 0.3`) discourages mode collapse where the agent converges to a single high-scoring molecule.
- **Novelty bonus** (`--novelty-bonus 0.1 --reference-file ref.smi`) rewards exploration away from known molecules.
- PPO trains faster than SAC on this task (on-policy rollouts fit sequential SMILES generation well); SAC can reach higher sample diversity with enough replay buffer.

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{druggen_rl,
  author  = {Kabeer, Rajul},
  title   = {DrugGen-RL: Reinforcement Learning for De Novo Drug Molecule Generation},
  year    = {2026},
  url     = {https://github.com/rajul-kk/Drug_Generation}
}
```
