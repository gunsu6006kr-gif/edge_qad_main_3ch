import torch
import torch.nn as nn
from .shuffleFAC import SelfAttention, FrequencyPositionalEncoding
import torch.nn.functional as F
from .shuffleFAC import shuffleFAC
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, fuse_modules

def kd_loss_logits(student_logits, teacher_logits, T: float):
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

def exclude_attention_from_qat(model):
    for m in model.modules():
        if isinstance(m, (SelfAttention, FrequencyPositionalEncoding)):
            m.qconfig = None 

def fuse_aqd_module(model):
    sub_model = model.cnn.cnn
    torch.ao.quantization.fuse_modules(sub_model, ['conv0','batchnorm0','relu0'], inplace=True)
    for i in range(1,7):
        layers_to_fuse = [
            f'point-wise conv{i}',
            f'batchnorm{i}',
            f'relu{i}'
        ]
        torch.ao.quantization.fuse_modules(sub_model, layers_to_fuse, inplace=True)
    return model

def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = None
    candidate_keys = ['model_state_dict', 'model_state', 'state_dict', 'model']
    for key in candidate_keys:
        if isinstance(checkpoint, dict) and key in checkpoint:
            print(f"Found weights under key: '{key}'")
            pretrained_dict = checkpoint[key]
            break
            
    if pretrained_dict is None:
        if any('weight' in k for k in checkpoint.keys()):
            print("Checkpoint seems to be the state_dict itself.")
            pretrained_dict = checkpoint
        else:
            print(f"\n[Error] Cannot find state_dict! Available keys: {list(checkpoint.keys())}")
            return

    model_dict = model.state_dict()
    new_state_dict = {}
    loaded_count = 0
    
    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            
        if k in model_dict:
            target_shape = model_dict[k].shape
            source_shape = v.shape
            
            if source_shape == target_shape:
                new_state_dict[k] = v
                loaded_count += 1
            elif len(source_shape) == len(target_shape):
                try:
                    slices = [slice(0, min(d_s, d_t)) for d_s, d_t in zip(source_shape, target_shape)]
                    new_state_dict[k] = v[slices]
                    loaded_count += 1
                except:
                    pass
    
    if loaded_count > 0:
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        print(f"✅ Successfully loaded {loaded_count} layers.")
    else:
        print("⚠️ No layers were loaded. Check if the model architecture matches.")

def build_qat_student(crnn_cfg, pretrained_path):
    m = Q_shuffleFAC(**crnn_cfg)
    print(m)
    if pretrained_path:
        load_pretrained_weights(m, pretrained_path)
    exclude_attention_from_qat(m)
    m.fc.qconfig = None
    torch.backends.quantized.engine = 'fbgemm'
    m.qconfig = get_default_qat_qconfig('fbgemm')
    m.eval()
    m = fuse_aqd_module(m)
    m.train()
    m = prepare_qat(m, inplace=True)
    return m

class Q_shuffleFAC(shuffleFAC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.flat = nn.Flatten()
    def forward(self, x, return_feats=False):
        x = x.transpose(2,3)
        x = self.quant(x)
        feature_map = self.cnn(x)
        fmap = self.dequant(feature_map)
        pooled = self.head(fmap)
        flat = self.flat(pooled)
        logits = self.fc(flat)

        if return_feats:
            return logits, fmap
        else:
            return logits