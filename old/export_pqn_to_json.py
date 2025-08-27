#!/usr/bin/env python3
"""
Export a PQN checkpoint to a browser-friendly JSON file for client-side inference.

Usage:
  uv run export_pqn_to_json.py \
    --checkpoint checkpoints/pqn_model_batch50_20250819_170514.pt \
    --out model/pqn_model_batch50.json
"""

import argparse
import json
from pathlib import Path

import torch

from pqn_model import PQN


def export(checkpoint_path: Path, out_path: Path):
    device = 'cpu'
    model = PQN(state_dim=104, action_dim=100, hidden_dims=(256, 256), use_layernorm=True)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    sd = model.state_dict()

    def tolist(t):
        return t.detach().cpu().numpy().tolist()

    js = {
        'state_dim': 104,
        'action_dim': 100,
        'hidden_dims': [256, 256],
        'layers': {
            'linear0': {
                'weight': tolist(sd['feature_layers.0.weight']),
                'bias': tolist(sd['feature_layers.0.bias'])
            },
            'linear1': {
                'weight': tolist(sd['feature_layers.2.weight']),
                'bias': tolist(sd['feature_layers.2.bias'])
            },
            'layernorm': {
                'gamma': tolist(sd['layer_norm.weight']),
                'beta': tolist(sd['layer_norm.bias']),
                'eps': model.layer_norm.eps
            },
            'output': {
                'weight': tolist(sd['output_layer.weight']),
                'bias': tolist(sd['output_layer.bias'])
            }
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(js, f)
    print(f"Exported model to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == '__main__':
    main()
