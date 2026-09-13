"""Evaluation entry point: unlearn a target class and score the result."""

import argparse

import yaml

from .dataset import setup_data
from .metrics import evaluate_model
from .model import load_pretrained
from .train import unlearn
from .utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Unlearn a target class and score the result.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--target-class", type=int, default=None, help="Override the target class from the config.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()

    target_class = args.target_class if args.target_class is not None else config["target_class"]

    model = load_pretrained(config["checkpoint_path"]).to(device)
    initial_state_dict = {name: param.detach().clone() for name, param in model.named_parameters()}

    forget_loader, retain_loader = setup_data(
        target_class, data_root=config["data_root"], batch_size=config["batch_size"]
    )

    unlearned_model = unlearn(
        model,
        forget_loader,
        retain_loader,
        steps=config["steps"],
        learning_rate=config["learning_rate"],
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        delta=config["delta"],
        device=device,
    )

    results = evaluate_model(unlearned_model, forget_loader, retain_loader, initial_state_dict, device=device)

    print(f"Target class:                  {target_class}")
    print(f"KL(forget || uniform):         {results['kl_to_uniform']:.4f}")
    print(f"Retain-set accuracy:           {results['acc_retain']:.4f}")
    print(f"L2 weight distance:            {results['l2_distance']:.4f}")
    print(f"Forget-set accuracy:           {results['acc_forget']:.4f}")
    print(f"Estimated task score (0-100):  {results['score']:.2f}")
    return results


if __name__ == "__main__":
    main()
