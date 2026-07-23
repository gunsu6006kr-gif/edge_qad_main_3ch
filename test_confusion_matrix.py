import argparse
import copy
import glob
import os
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_preprocessing import dataset
from src.shuffleFAC import shuffleFAC


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a best_model_ checkpoint on the test set and save confusion matrix images."
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="/home/user/Desktop/data/ori_DSOS_data_4ch_cache/v1_test_4ch_cache.pt",
        help="Path to the 4-channel test cache or PT list.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/best_model_0722_021706.pt",
        help="Path to a best_model_*.pt checkpoint. If omitted, the newest checkpoint is used.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory used when --checkpoint_path is omitted.",
    )
    parser.add_argument(
        "--use_channels",
        type=str,
        default="0,3",
        help="Channels to use from pt file. 3 is the time-frequency mixed second derivative. Example: 0,1,2,3",
    )
    parser.add_argument("--scale", type=int, default=1, help="ShuffleFAC scale used during training.")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for test evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Config YAML path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/confusion_matrix/best_model_0722_021706",
        help="Directory to save confusion matrix outputs.",
    )
    return parser.parse_args()


def find_latest_best_model(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "best_model_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found with pattern: {pattern}")
    return max(candidates, key=os.path.getmtime)


def resolve_checkpoint_path(checkpoint_path, checkpoint_dir):
    if checkpoint_path is None:
        return find_latest_best_model(checkpoint_dir)

    if os.path.exists(checkpoint_path):
        return checkpoint_path

    print(f"[WARN] Checkpoint not found: {checkpoint_path}")
    latest_checkpoint = find_latest_best_model(checkpoint_dir)
    print(f"[WARN] Falling back to latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def build_student(configs, n_in_channel, scale):
    crnn_cfg = copy.deepcopy(configs["student"])
    crnn_cfg.pop("n_in_channel", None)
    crnn_cfg["n_input_ch"] = n_in_channel

    if "nb_filters" in crnn_cfg:
        crnn_cfg["nb_filters"] = [int(x / scale) for x in crnn_cfg["nb_filters"]]

    return shuffleFAC(**crnn_cfg)


def clean_state_dict_for_eval(state_dict):
    ignored_suffixes = ("total_ops", "total_params")
    return {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(ignored_suffixes)
    }


def checkpoint_name_for_filename(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    checkpoint_name = os.path.splitext(checkpoint_name)[0]
    return "".join(char if char.isalnum() or char in ("-", "_", ",") else "_" for char in checkpoint_name)


def get_checkpoint_input_channels(state_dict):
    conv0_weight = state_dict.get("cnn.cnn.conv0.weight")
    if conv0_weight is None:
        return None
    return int(conv0_weight.shape[1])


@torch.no_grad()
def predict_test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(test_loader, total=len(test_loader), desc="Test", leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        pred = torch.argmax(outputs, dim=1)

        total_loss += loss.item()
        num_batches += 1
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if not y_true_all:
        raise RuntimeError("Test loader produced no batches.")

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    avg_loss = total_loss / max(1, num_batches)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return avg_loss, acc, macro_f1, y_true, y_pred


def plot_confusion_matrix(cm, class_names, title, save_path, fmt="d"):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_outputs(y_true, y_pred, class_names, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    raw_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_raw.png")
    norm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.png")
    csv_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_raw.csv")

    plot_confusion_matrix(cm, class_names, "Test Confusion Matrix", raw_path, fmt="d")
    plot_confusion_matrix(cm_normalized, class_names, "Test Confusion Matrix (Normalized)", norm_path, fmt=".2f")
    np.savetxt(csv_path, cm, fmt="%d", delimiter=",", header=",".join(class_names), comments="")

    return raw_path, norm_path, csv_path


def main():
    args = get_args()
    use_channels = [int(x) for x in args.use_channels.split(",")]
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, args.checkpoint_dir)

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = dataset(args.test_list, mel_kwargs=configs["feats"], use_channels=use_channels)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_student(configs, n_in_channel=len(use_channels), scale=args.scale).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state_dict = clean_state_dict_for_eval(state["model_state_dict"])
    checkpoint_input_channels = get_checkpoint_input_channels(model_state_dict)
    if checkpoint_input_channels is not None and checkpoint_input_channels != len(use_channels):
        raise ValueError(
            f"Checkpoint expects {checkpoint_input_channels} input channel(s), "
            f"but --use_channels={args.use_channels} makes {len(use_channels)} channel(s). "
            "Use the same --use_channels setting that was used during training."
        )
    model.load_state_dict(model_state_dict)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_macro_f1, y_true, y_pred = predict_test(model, test_loader, criterion, device)

    class_names = [name for name, _ in sorted(test_set.class_name_to_id.items(), key=lambda item: item[1])]
    time_str = datetime.now().strftime("%m%d_%H%M%S")
    checkpoint_tag = checkpoint_name_for_filename(checkpoint_path)
    prefix = f"{checkpoint_tag}_test_{time_str}"
    raw_path, norm_path, csv_path = save_confusion_outputs(y_true, y_pred, class_names, args.output_dir, prefix)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")
    print(f"Saved raw confusion matrix image       : {raw_path}")
    print(f"Saved normalized confusion matrix image: {norm_path}")
    print(f"Saved raw confusion matrix csv         : {csv_path}")


if __name__ == "__main__":
    main()
