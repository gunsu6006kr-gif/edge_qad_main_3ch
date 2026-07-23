from sys import breakpointhook
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.transforms import RandomApply
import torchaudio

def kd_loss_logits(student_logits, teacher_logits, T: float):
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

def prepare(features):
    if features.dim() == 4:          # [B, C, F, T] 형태
        B, C, F, T = features.shape
        features = features.view(B, C, -1).mean(dim=2)  # F·T 축 평균 → [B, C]
    elif features.dim() == 3:        # [B, T, D] 형태
        features = features.mean(dim=1)                 # T 축 평균 → [B, D]
    elif features.dim() == 2:
        pass                          # 이미 [B, D]
    else:
        raise ValueError("지원하지 않는 텐서 차원입니다.")
    return features

def loss_MVD(discriminator, teacher_feats, student_feats):
    # Discriminator loss: classify teacher features as real, student features as fake
    teacher_feats = prepare(teacher_feats)
    student_feats = prepare(student_feats)
    logits_teacher = discriminator(teacher_feats.detach())
    logits_student = discriminator(student_feats.detach())
    loss_teacher = F.binary_cross_entropy_with_logits(
        logits_teacher, torch.ones_like(logits_teacher))
    loss_student = F.binary_cross_entropy_with_logits(
        logits_student, torch.zeros_like(logits_student))
    return 0.5 * (loss_teacher + loss_student)

def loss_mvg(discriminator, h_s_prime):
    feats = prepare(h_s_prime)
    logits = discriminator(feats)
    targets = torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets)

def loss_proj2(h_T, h_s_prime):
    '''두 개의 텐서간의 거리 유사도를 계산'''
    h_T = F.adaptive_avg_pool2d(h_T, (1,1))
    return F.mse_loss(h_s_prime, h_T)

def cosine_proj(cosine_loss, h_T, h_s_prime):
    h_T = prepare(h_T)
    h_s_prime = prepare(h_s_prime)
    target = torch.ones(h_T.size(0), device=h_T.device)
    output = cosine_loss(h_T, h_s_prime, target)
    return output

    

def apply_specaugment(batch_x, training=True, t_l=5, t_p=0.2):
    if training:
        timemask = torchaudio.transforms.TimeMasking(t_l, True, t_p)
        return timemask(batch_x.transpose(1, -1)).transpose(1, -1)
    return batch_x


class Discriminator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim1=384, hidden_dim2=192):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, features):
        # if features.dim() > 2:
            # features = features.view(features.size(0), -1)
        return self.model(features)

class GL_Projector(nn.Module):
    def __init__(self, in_channels=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
    def forward(self, h_S):
        x = self.proj(h_S)
        x = self.norm(x)
        return x

