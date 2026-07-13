# Multi-Robot Ackermann Navigation with TD3 and Interaction-Aware RVO Features

This repository contains the core implementation of a decentralized low-level
multi-robot navigation policy for Ackermann-steered robots.

## Included Components

- Low-level multi-robot Ackermann environment
- TD3-based continuous control policy
- Interaction-aware RVO feature construction
- Curriculum training code
- Configuration file

## Not Included

The following files are intentionally excluded:

- trained checkpoints / model weights
- visualization scripts
- ablation experiment scripts
- baseline comparison scripts
- logs, result CSVs, figures, and GIFs

## Main Training Entry

```bash
python -m tests.train_low_curriculum_v2

