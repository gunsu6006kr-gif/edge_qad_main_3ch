import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CLASS_NAMES = ["Cargo", "Passengership", "Tanker", "Tug", "Nontarget"]


def resolve_existing_path(path):
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)

    script_dir = Path(__file__).resolve().parent
    script_dir_candidate = script_dir / path
    if script_dir_candidate.exists():
        return str(script_dir_candidate)

    parts = candidate.parts
    if "edge-qad-main-3ch" in parts:
        repo_idx = parts.index("edge-qad-main-3ch")
        repo_relative = Path(*parts[repo_idx + 1 :])
        repo_relative_candidate = script_dir / repo_relative
        if repo_relative_candidate.exists():
            return str(repo_relative_candidate)

    return path


def load_confusion_csv(path):
    path = resolve_existing_path(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty csv: {path}")

    header = rows[0]
    data_rows = rows[1:]

    if all(cell.strip().lstrip("-").isdigit() for cell in header):
        class_names = DEFAULT_CLASS_NAMES[: len(header)]
        data_rows = rows
    else:
        class_names = [cell.strip() for cell in header]

    matrix = np.array([[int(cell) for cell in row] for row in data_rows], dtype=np.int64)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix must be square: {path}, shape={matrix.shape}")

    if len(class_names) != matrix.shape[0]:
        raise ValueError(
            f"Class count and matrix shape mismatch: {path}, classes={len(class_names)}, shape={matrix.shape}"
        )

    return class_names, matrix


def row_normalize(matrix):
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum != 0)


def save_matrix_csv(path, class_names, matrix, fmt):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)

        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + [format(value, fmt) for value in row])


def plot_matrix(matrix, class_names, title, save_path, fmt, cmap="RdBu_r", center_zero=True):
    fig, ax = plt.subplots(figsize=(9, 7))

    if center_zero:
        abs_max = float(np.max(np.abs(matrix))) if matrix.size else 1.0
        abs_max = abs_max if abs_max > 0 else 1.0
        im = ax.imshow(matrix, cmap=cmap, vmin=-abs_max, vmax=abs_max)
    else:
        im = ax.imshow(matrix, cmap=cmap)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = np.max(np.abs(matrix)) * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if abs(value) > threshold else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_advantage_matrix(original, derivative):
    raw_diff = original - derivative
    advantage = raw_diff.copy()

    for idx in range(advantage.shape[0]):
        advantage[idx, idx] = derivative[idx, idx] - original[idx, idx]

    return advantage


def write_class_summary(path, class_names, original, derivative):
    orig_norm = row_normalize(original)
    deriv_norm = row_normalize(derivative)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class",
                "total",
                "original_correct",
                "derivative_correct",
                "correct_delta_derivative_minus_original",
                "original_recall",
                "derivative_recall",
                "recall_delta_derivative_minus_original",
                "most_reduced_confusion_pred",
                "most_reduced_confusion_count_original_minus_derivative",
                "most_increased_confusion_pred",
                "most_increased_confusion_count_original_minus_derivative",
            ]
        )

        for i, class_name in enumerate(class_names):
            off_diag_diff = original[i] - derivative[i]
            off_diag_diff[i] = 0

            reduced_idx = int(np.argmax(off_diag_diff))
            increased_idx = int(np.argmin(off_diag_diff))

            writer.writerow(
                [
                    class_name,
                    int(original[i].sum()),
                    int(original[i, i]),
                    int(derivative[i, i]),
                    int(derivative[i, i] - original[i, i]),
                    f"{orig_norm[i, i]:.6f}",
                    f"{deriv_norm[i, i]:.6f}",
                    f"{deriv_norm[i, i] - orig_norm[i, i]:.6f}",
                    class_names[reduced_idx],
                    int(off_diag_diff[reduced_idx]),
                    class_names[increased_idx],
                    int(off_diag_diff[increased_idx]),
                ]
            )


def write_pair_summary(path, class_names, original, derivative, top_k):
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            raw_diff = int(original[i, j] - derivative[i, j])

            if i == j:
                derivative_advantage = int(derivative[i, j] - original[i, j])
                meaning = "correct increased by derivative" if derivative_advantage > 0 else "correct decreased by derivative"
            else:
                derivative_advantage = raw_diff
                meaning = "confusion reduced by derivative" if derivative_advantage > 0 else "confusion increased by derivative"

            rows.append(
                {
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "original": int(original[i, j]),
                    "derivative": int(derivative[i, j]),
                    "original_minus_derivative": raw_diff,
                    "derivative_advantage": derivative_advantage,
                    "meaning": meaning,
                }
            )

    rows.sort(key=lambda row: abs(row["derivative_advantage"]), reverse=True)

    if top_k > 0:
        rows = rows[:top_k]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "true_class",
                "pred_class",
                "original",
                "derivative",
                "original_minus_derivative",
                "derivative_advantage",
                "meaning",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_top_advantages(class_names, advantage, top_k):
    flat = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            value = int(advantage[i, j])
            if value == 0:
                continue

            if i == j:
                desc = f"{true_name}: correct predictions changed"
            else:
                desc = f"{true_name} -> {pred_name}: confusion changed"

            flat.append((abs(value), value, desc))

    flat.sort(reverse=True)

    print("\nTop derivative advantages (+ means derivative is better):")
    for _, value, desc in flat[:top_k]:
        print(f"  {value:+d}  {desc}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare original and derivative confusion matrices."
    )
    parser.add_argument(
        "--original_csv",
        type=str,
        default="./exp/confusion_matrix/0_1ch_best_test_0701_120208_confusion_matrix_raw.csv",
        help="Raw confusion matrix csv from original/log-mel model.",
    )
    parser.add_argument(
        "--derivative_csv",
        type=str,
        default="./exp/confusion_matrix/0,2_2ch_best_test_0701_115712_confusion_matrix_raw.csv",
        help="Raw confusion matrix csv from derivative/3ch model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/confusion_matrix_compare",
        help="Directory to save comparison outputs.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Output filename tag.")
    parser.add_argument("--top_k", type=int, default=25, help="Rows to save in pair summary. 0 saves all.")

    return parser.parse_args()


def main():
    args = get_args()

    original_classes, original = load_confusion_csv(args.original_csv)
    derivative_classes, derivative = load_confusion_csv(args.derivative_csv)

    if original_classes != derivative_classes:
        raise ValueError(
            f"Class order mismatch:\noriginal={original_classes}\nderivative={derivative_classes}"
        )

    if original.shape != derivative.shape:
        raise ValueError(f"Matrix shape mismatch: original={original.shape}, derivative={derivative.shape}")

    class_names = original_classes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or datetime.now().strftime("%m%d_%H%M%S")

    raw_diff = original - derivative
    norm_diff = row_normalize(original) - row_normalize(derivative)
    advantage = make_advantage_matrix(original, derivative)
    norm_advantage = make_advantage_matrix(row_normalize(original), row_normalize(derivative))

    save_matrix_csv(output_dir / f"{tag}_original_minus_derivative_raw.csv", class_names, raw_diff, "d")
    save_matrix_csv(output_dir / f"{tag}_original_minus_derivative_normalized.csv", class_names, norm_diff, ".6f")
    save_matrix_csv(output_dir / f"{tag}_derivative_advantage_raw.csv", class_names, advantage, "d")
    save_matrix_csv(output_dir / f"{tag}_derivative_advantage_normalized.csv", class_names, norm_advantage, ".6f")

    plot_matrix(
        raw_diff,
        class_names,
        "Original - Derivative (Raw)",
        output_dir / f"{tag}_original_minus_derivative_raw.png",
        "d",
    )
    plot_matrix(
        norm_diff,
        class_names,
        "Original - Derivative (Normalized)",
        output_dir / f"{tag}_original_minus_derivative_normalized.png",
        ".2f",
    )
    plot_matrix(
        advantage,
        class_names,
        "Derivative Advantage (Raw)",
        output_dir / f"{tag}_derivative_advantage_raw.png",
        "d",
    )
    plot_matrix(
        norm_advantage,
        class_names,
        "Derivative Advantage (Normalized)",
        output_dir / f"{tag}_derivative_advantage_normalized.png",
        ".2f",
    )

    write_class_summary(output_dir / f"{tag}_class_summary.csv", class_names, original, derivative)
    write_pair_summary(output_dir / f"{tag}_pair_summary.csv", class_names, original, derivative, args.top_k)

    print(f"Original csv  : {args.original_csv}")
    print(f"Derivative csv: {args.derivative_csv}")
    print(f"Output dir    : {output_dir}")
    print_top_advantages(class_names, advantage, min(args.top_k, 10) if args.top_k != 0 else 10)


if __name__ == "__main__":
    main()
