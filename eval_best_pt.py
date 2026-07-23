import argparse
import copy
import csv
import glob
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_preprocessing import dataset
from src.shuffleFAC import shuffleFAC


CLASS_NAMES = ["Cargo", "Passengership", "Tanker", "Tug", "Nontarget"]


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a best.pt checkpoint on the test set.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/no_change_feature_0,2.pt",
        help="Path to checkpoint pt file.",
    )
    parser.add_argument(
        "--checkpoint_paths",
        nargs="+",
        default=None,
        help="Optional multiple checkpoint paths or glob patterns for mean/std evaluation.",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="/home/user/Desktop/data/DSOS_pt/DSOS_data_3ch_pt/cache/v1_test_3ch_cache.pt",
        help="Path to test txt list or split cache pt.",
    )
    parser.add_argument(
        "--use_channels",
        type=str,
        default="0,2",
        help="Channels to use. Channel 3 is the time-frequency mixed second derivative. Examples: 0 / 0,3 / 0,1,2,3",
    )
    parser.add_argument("--scale", type=int, default=1, help="ShuffleFAC scale used during training.")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for test evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Config YAML path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/test_results",
        help="Directory to save csv outputs.",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save per-sample predictions and confusion matrix csv files.",
    )
    return parser.parse_args()


def parse_channels(use_channels):
    return [int(x.strip()) for x in use_channels.split(",") if x.strip()]


def resolve_checkpoint_paths(args):
    paths = args.checkpoint_paths if args.checkpoint_paths else [args.checkpoint_path]
    resolved = []

    for path in paths:
        matches = sorted(glob.glob(path)) if glob.has_magic(path) else [path]
        resolved.extend(matches)

    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        raise FileNotFoundError("No checkpoint paths matched.")

    return resolved


def build_student(configs, n_in_channel, scale):
    crnn_cfg = copy.deepcopy(configs["student"])
    crnn_cfg.pop("n_in_channel", None)
    crnn_cfg["n_input_ch"] = n_in_channel

    if "nb_filters" in crnn_cfg:
        crnn_cfg["nb_filters"] = [int(x / scale) for x in crnn_cfg["nb_filters"]]

    return shuffleFAC(**crnn_cfg)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if key in checkpoint:
            return checkpoint[key]

    if any("weight" in key for key in checkpoint.keys()):
        return checkpoint

    raise KeyError(f"Cannot find model state dict. Available keys: {list(checkpoint.keys())}")


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.endswith(("total_ops", "total_params")):
            continue
        cleaned[key] = value
    return cleaned


def get_checkpoint_input_channels(state_dict):
    conv0_weight = state_dict.get("cnn.cnn.conv0.weight")
    if conv0_weight is None:
        return None
    return int(conv0_weight.shape[1])


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(loader, total=len(loader), desc="Test", leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        pred = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        num_batches += 1
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if not y_true_all:
        raise RuntimeError("Test loader produced no batches.")

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    avg_loss = total_loss / max(1, num_batches)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    return avg_loss, acc, macro_f1, weighted_f1, y_true, y_pred


def save_outputs(output_dir, checkpoint_path, y_true, y_pred, class_names):
    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.now().strftime("%m%d_%H%M%S")
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    prefix = f"{checkpoint_name}_test_{time_str}"

    pred_csv = os.path.join(output_dir, f"{prefix}_predictions.csv")
    cm_csv = os.path.join(output_dir, f"{prefix}_confusion_matrix_raw.csv")
    cm_norm_csv = os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.csv")

    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_id", "true_name", "pred_id", "pred_name", "correct"])
        for idx, (true_id, pred_id) in enumerate(zip(y_true, y_pred)):
            writer.writerow(
                [
                    idx,
                    int(true_id),
                    class_names[int(true_id)],
                    int(pred_id),
                    class_names[int(pred_id)],
                    int(true_id == pred_id),
                ]
            )

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",", header=",".join(class_names), comments="")
    np.savetxt(cm_norm_csv, cm_norm, fmt="%.6f", delimiter=",", header=",".join(class_names), comments="")

    return pred_csv, cm_csv, cm_norm_csv


def main():
    args = get_args()
    use_channels = parse_channels(args.use_channels)
    checkpoint_paths = resolve_checkpoint_paths(args)

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoints: {len(checkpoint_paths)}")
    print(f"Test list: {args.test_list}")
    print(f"Use channels: {use_channels}")

    test_set = dataset(args.test_list, mel_kwargs=configs["feats"], use_channels=use_channels)
    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    criterion = nn.CrossEntropyLoss()
    class_names = [
        name for name, _ in sorted(test_set.class_name_to_id.items(), key=lambda item: item[1])
    ]

    results = []

    for checkpoint_path in checkpoint_paths:
        print("\n" + "=" * 60)
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = clean_state_dict(extract_state_dict(checkpoint))

        checkpoint_input_channels = get_checkpoint_input_channels(state_dict)
        if checkpoint_input_channels is not None and checkpoint_input_channels != len(use_channels):
            raise ValueError(
                f"Checkpoint expects {checkpoint_input_channels} input channel(s), "
                f"but --use_channels={args.use_channels} makes {len(use_channels)} channel(s)."
            )

        model = build_student(configs, n_in_channel=len(use_channels), scale=args.scale).to(device)
        model.load_state_dict(state_dict)

        test_loss, test_acc, macro_f1, weighted_f1, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        results.append(
            {
                "checkpoint": checkpoint_path,
                "loss": test_loss,
                "acc": test_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }
        )

        print(f"loss       : {test_loss:.4f}")
        print(f"acc        : {test_acc:.4f}")
        print(f"macro_f1   : {macro_f1:.4f}")
        print(f"weighted_f1: {weighted_f1:.4f}")

        print("\nClassification report")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        labels = list(range(len(class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

        print("Confusion matrix raw")
        print(cm)
        print("\nConfusion matrix normalized")
        print(np.round(cm_norm, 4))

        if args.save_csv:
            pred_csv, cm_csv, cm_norm_csv = save_outputs(
                args.output_dir,
                checkpoint_path,
                y_true,
                y_pred,
                class_names,
            )
            print("\nSaved outputs")
            print(f"predictions: {pred_csv}")
            print(f"cm raw     : {cm_csv}")
            print(f"cm norm    : {cm_norm_csv}")

    losses = np.array([result["loss"] for result in results], dtype=float)
    accs = np.array([result["acc"] for result in results], dtype=float)
    macro_f1s = np.array([result["macro_f1"] for result in results], dtype=float)
    weighted_f1s = np.array([result["weighted_f1"] for result in results], dtype=float)

    print("\n" + "=" * 60)
    print(f"SUMMARY ({len(results)} checkpoint(s), mean ± std)")
    print("=" * 60)
    print(f"loss       : {losses.mean():.4f} ± {losses.std():.4f}")
    print(f"acc        : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"macro_f1   : {macro_f1s.mean():.4f} ± {macro_f1s.std():.4f}")
    print(f"weighted_f1: {weighted_f1s.mean():.4f} ± {weighted_f1s.std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
