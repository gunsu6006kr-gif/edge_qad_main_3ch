set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
# ShuffleFAC V3
python train.py --mode 'ad' --lr 0.01 --lr_D 0.005 --epoch 150 --batch_size 48 --scale 1 --data_root '../deepship_data/DeepShip/preprocessed_data' \
 --pretrained_path './checkpoints/ref_ShuffleFAC_V3.pt'
#ShuffleFAC V2
python train.py --mode 'ad' --lr 0.01 --lr_D 0.005 --epoch 150 --batch_size 48 --scale 2 --data_root '../deepship_data/Deepship/preprocessed_data' \
 --pretrained_path './checkpoints/ref_ShuffleFAC_V2.pt'
#ShuffleFAC V1
python train.py --mode 'ad' --lr 0.01 --lr_D 0.005 --epoch 150 --batch_size 48 --scale 4 --data_root '../deepship_data/Deepship/preprocessed_data' \
 --pretrained_path './checkpoints/ref_ShuffleFAC_V1.pt'