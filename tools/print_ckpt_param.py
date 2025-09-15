import os
import sys

import torch


def print_ckpt_param(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    o = open(ckpt_path.replace('.pth', '.txt'), 'w')
    for key in state.keys():
        if key == 'model':
            for k in state[key].keys():
                o.write(f"{key} : {k} {state[key][k].shape}\n")
        else:
            if isinstance(state[key], dict):
                for k in state[key].keys():
                    o.write(f"{key} : {k} {state[key][k].shape}\n")
            elif isinstance(state[key], list):
                for k in state[key]:
                    o.write(f"{key} : {k}\n")
    o.close()

if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    print_ckpt_param(ckpt_path)