import sys
import os

# ──────────────────────────────────────────────────
# 파이썬 경로(Path) 강제 추가 (ModuleNotFoundError 해결용)
# ──────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
import yaml
import copy

from src.shuffleFAC import shuffleFAC
# 고객님이 작성하신 경로에 맞춤 (data 폴더 안의 data_preprocessing.py)
from data.data_preprocessing import dataset as AudioDataset

# ──────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REAL_ROOT  = "/home/user/Desktop/data/preprocessed_data/train"
DIFF_ROOT  = "/home/user/Desktop/data/data_diff/train"

CKPT_PATH  = "./checkpoints/FAC_best.pt"   # 실제 체크포인트 이름으로 변경하세요!
YAML_PATH  = "./default.yaml"

# [중요 수정] 파일이 실제로 존재하는 경로로 수정! (baek -> user)
STATS_PATH = "/home/user/Desktop/data/stats.pt" 

BATCH_SIZE = 64
N_SAMPLES  = None

CLASS_NAMES = {0: "Cargo", 1: "Passengership", 2: "Tanker", 3: "Tug"}

# [중요 복구] AudioDataset을 작동시키기 위해 반드시 필요한 설정값
FEATS_CFG = {
    'n_mels': 128,
    'n_fft': 4096,
    'hop_length': 2048,
    'win_length': 4096,
    'sample_rate': 16000,
    'f_min': 0.0,
    'f_max': 8000,
    'power': 1.0
}

# ──────────────────────────────────────────────────
# 정규화
# ──────────────────────────────────────────────────
# 보안 경고를 없애고 올바른 파일을 로드하도록 weights_only=True 추가
stats = torch.load(STATS_PATH, weights_only=True)
MEAN  = stats["mean"]
STD   = stats["std"]

def normalize_mel(mel):
    mel = (mel - MEAN) / STD
    mel = mel.clamp(-3, 3) / 3
    return mel

# ──────────────────────────────────────────────────
# 임베딩 추출용 모델 래퍼
# ──────────────────────────────────────────────────
class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, fmap = self.model(x, return_feats=True)
        emb = fmap.view(fmap.size(0), -1)
        return emb

# ──────────────────────────────────────────────────
# 임베딩 추출 함수
# ──────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(extractor, loader, device, n_samples=None):
    embeddings = []
    count = 0

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        emb = extractor(batch_x)
        embeddings.append(emb.cpu().numpy())
        count += batch_x.shape[0]

        if n_samples is not None and count >= n_samples:
            break

    embeddings = np.concatenate(embeddings, axis=0)
    if n_samples is not None:
        embeddings = embeddings[:n_samples]
    return embeddings

# ──────────────────────────────────────────────────
# FAD 계산 (Frechet distance)
# ──────────────────────────────────────────────────
def calculate_fad(real_embeddings, test_embeddings, eps=1e-6):
    mu1 = np.mean(real_embeddings, axis=0)
    mu2 = np.mean(test_embeddings, axis=0)

    sigma1 = np.cov(real_embeddings, rowvar=False)
    sigma2 = np.cov(test_embeddings, rowvar=False)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fad

# ──────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────
def main():
    with open(YAML_PATH, "r") as f:
        configs = yaml.safe_load(f)
    crnn_cfg = copy.deepcopy(configs["student"]) 

    model = shuffleFAC(**crnn_cfg).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    
    # KeyError 방지용 안전 코드
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"분류 모델 로드 완료: {CKPT_PATH}")

    extractor = EmbeddingExtractor(model).to(DEVICE)
    extractor.eval()

    # [중요 수정] PreloadedMelDataset을 전부 지우고 AudioDataset으로 교체 완료
    real_set = AudioDataset(file_path=REAL_ROOT, transform=normalize_mel, mel_kwargs=FEATS_CFG)
    diff_set = AudioDataset(file_path=DIFF_ROOT, transform=normalize_mel, mel_kwargs=FEATS_CFG)

    real_loader = DataLoader(real_set, batch_size=BATCH_SIZE, shuffle=True)
    diff_loader = DataLoader(diff_set, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Real samples : {len(real_set)}")
    print(f"Diff samples : {len(diff_set)}")

    print("\n실제 데이터 임베딩 추출 중...")
    real_emb = extract_embeddings(extractor, real_loader, DEVICE, N_SAMPLES)
    print(f"  shape: {real_emb.shape}")

    print("미분(Diff) 데이터 임베딩 추출 중...")
    diff_emb = extract_embeddings(extractor, diff_loader, DEVICE, N_SAMPLES)
    print(f"  shape: {diff_emb.shape}")

    fad_score = calculate_fad(real_emb, diff_emb)
    print(f"\n{'='*50}")
    print(f"Overall FAD (Real vs Diff): {fad_score:.4f}")
    print(f"{'='*50}")

    print("\n클래스별 FAD:")
    for cls_id, cls_name in CLASS_NAMES.items():
        real_cls_set  = AudioDataset(
            file_path=os.path.join(REAL_ROOT, cls_name), transform=normalize_mel, mel_kwargs=FEATS_CFG
        )
        diff_cls_set = AudioDataset(
            file_path=os.path.join(DIFF_ROOT, cls_name), transform=normalize_mel, mel_kwargs=FEATS_CFG
        )

        if len(real_cls_set) == 0 or len(diff_cls_set) == 0:
            print(f"  [{cls_name:15s}] FAD = 데이터 없음 (건너뜀)")
            continue

        real_cls_loader = DataLoader(real_cls_set,  batch_size=BATCH_SIZE, shuffle=True)
        diff_cls_loader = DataLoader(diff_cls_set, batch_size=BATCH_SIZE, shuffle=True)

        real_cls_emb = extract_embeddings(extractor, real_cls_loader,  DEVICE, N_SAMPLES)
        diff_cls_emb = extract_embeddings(extractor, diff_cls_loader, DEVICE, N_SAMPLES)

        cls_fad = calculate_fad(real_cls_emb, diff_cls_emb)
        print(f"  [{cls_name:15s}] FAD = {cls_fad:.4f}")

    return fad_score

if __name__ == "__main__":
    main()