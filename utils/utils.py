import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import soundfile
import torch
"from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics"
from sed_scores_eval.base_modules.scores import create_score_dataframe
from thop import clever_format, profile

def calculate_macs(model, device, config, dataset=None):
    """
    The function calculate the multiply–accumulate operation (MACs) of the model given as input.

    Args:
        model: deep learning model to calculate the macs for
        config: config used to train the model
        dataset: dataset used to train the model

    Returns:

    """
    n_frames = int(
        (
            (config["feats"]["sample_rate"] * config["data"]["audio_max_len"])
            / config["feats"]["hop_length"]
        )
        + 1
    )
    input_size = [1,config["student"]["n_input_ch"], config["feats"]['n_mels'],n_frames] #[batch_size, channel, Time, Frequency]
    input = torch.randn(input_size).to(device)
    model.to(device)
    if "use_embeddings" in config["student"] and config["student"]["use_embeddings"]:
        audio, label, padded_indxs, path, embeddings = dataset[0]
        embeddings = embeddings.repeat(1, 1, 1)
        macs, params = profile(model, inputs=(input, None, embeddings))
    else:
        macs, params = profile(model, inputs=(input,))

    macs, params = clever_format([macs, params], "%.3f")
    return macs, params

def count_parameters(model):
    """
    모델의 파라미터 수를 계산합니다.
    
    Args:
        model: PyTorch 모델
    
    Returns:
        total_params: 총 파라미터 수
        trainable_params: 훈련 가능한 파라미터 수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

