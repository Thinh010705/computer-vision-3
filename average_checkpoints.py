import argparse
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Create a model soup by averaging compatible checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="./models/model_soup.pth")
    return parser.parse_args()


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    return checkpoint, state


def main():
    args = parse_args()
    checkpoints = []
    states = []
    for path in args.checkpoints:
        checkpoint, state = load_checkpoint(path)
        checkpoints.append(checkpoint)
        states.append(state)

    keys = list(states[0].keys())
    if any(list(state.keys()) != keys for state in states[1:]):
        raise ValueError("Checkpoints do not have identical model parameters.")

    averaged = OrderedDict()
    for key in keys:
        values = [state[key] for state in states]
        if values[0].dtype.is_floating_point:
            averaged[key] = torch.stack([value.float() for value in values]).mean(dim=0).to(values[0].dtype)
        else:
            averaged[key] = values[0].clone()

    maps = [
        float(checkpoint.get("mAP", 0.0))
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict)
    ]
    output = {
        "model_state_dict": averaged,
        "source_checkpoints": args.checkpoints,
        "source_maps": maps,
    }
    torch.save(output, args.output)
    print(f"Saved averaged checkpoint from {len(states)} models to {args.output}")


if __name__ == "__main__":
    main()
