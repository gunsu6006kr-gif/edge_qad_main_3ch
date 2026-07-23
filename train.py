from math import gamma
import sys
import torch
import torch.nn as nn
import os, csv, random, copy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import argparse
from src.FAC import CRNN
from src.shuffleFAC import shuffleFAC
from src.models import build_qat_student, load_pretrained_weights
from src.losses import GL_Projector, Discriminator
from src.train_engine import train, kd_train, ad_train, valid, test

from data.data_preprocessing import dataset
from utils.utils import calculate_macs, count_parameters

from codecarbon import OfflineEmissionsTracker
from datetime import datetime
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
from numpy import dtype

def get_args():
    parser = argparse.ArgumentParser(description="DeepShip Training Script")
    parser.add_argument(
        '--train_list',
        type=str,
        default='/home/user/Desktop/data/ori_DSOS_data_4ch_cache/v1_train_4ch_cache.pt',
        help='Path to train wav/pt list'
    )

    parser.add_argument(
        '--val_list',
        type=str,
        default='/home/user/Desktop/data/ori_DSOS_data_4ch_cache/v1_val_4ch_cache.pt',
        help='Path to validation wav/pt list'
    )

    parser.add_argument(
        '--test_list',
        type=str,
        default='/home/user/Desktop/data/ori_DSOS_data_4ch_cache/v1_test_4ch_cache.pt',
        help='Path to test wav/pt list'
    )

    parser.add_argument(
        '--use_channels',
        type=str,
        default='0,3',
        help='Channels to use. 0=LogMel, 1=Frequency, 2=Time, 3=Time-Frequency mixed second derivative'
    )
    # Set Train Mode(ref, kd, ad, qat, qad)
    parser.add_argument('--mode', type=str, default='ref', 
                        choices=['ref', 'kd', 'ad', 'qat', 'qad'], 
                        help='Training mode: ref, kd (Knowledge Distillation), ad (Adversarial Distillation), qat(Quantization Aware Training),' \
                        'qad(Quantization Adversarial Distillation)')

    # Set Train Hyper-Parameter
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--lr_D', type=float, default=0.005, help='Discriminator Learning Rate')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
    
    parser.add_argument('--scale', type=int, default=1, help='scale=1 : ShuffleFAC V3, scale=2 : ShuffleFAC V2,' \
    'scale=4 : ShuffleFAC V1')
    
    # Set Dataset & Pretrained path
    parser.add_argument('--data_root', type=str, default='/home/user/Desktop/DSOS/list/major', help='Path to dataset')
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/ref_ShuffleFAC_V3.pt')
    
    return parser.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    args = get_args()

    use_channels = [int(x) for x in args.use_channels.split(",")]
    n_in_channel = len(use_channels)

    print(f"Using channels: {use_channels}")
    print(f"Input channel count: {n_in_channel}")

    with open('./configs/default.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Training will run on CPU.")
    torch.backends.quantized.engine = 'fbgemm'
    print(torch.backends.quantized.engine)

    crnn_cfg = configs['student']
    crnn_cfg = copy.deepcopy(configs["student"])

    # shuffleFAC는 n_in_channel이 아니라 n_input_ch를 사용함
    crnn_cfg.pop("n_in_channel", None)
    crnn_cfg["n_input_ch"] = n_in_channel

    # calculate_macs용 configs도 같이 맞춤
    configs["student"].pop("n_in_channel", None)
    configs["student"]["n_input_ch"] = n_in_channel

    if 'nb_filters' in crnn_cfg:
        original_filters = crnn_cfg['nb_filters'] # [32, 64, 128, 256, 256, 256, 256]
        
        scaled_filters = [int(x / args.scale) for x in original_filters]
        
        crnn_cfg['nb_filters'] = scaled_filters
        
        print(f"Original nb_filters: {original_filters}")
        print(f"Scaled nb_filters  : {scaled_filters}")
    else:
        print("!!! Warning: 'nb_filters' key not found in YAML !!!")
    teacher_cfg = configs['teacher']
    feats_cfg = configs['feats']

    train_set = dataset(
        args.train_list,
        mel_kwargs=feats_cfg,
        use_channels=use_channels
    )

    val_set = dataset(
        args.val_list,
        mel_kwargs=feats_cfg,
        use_channels=use_channels
    )

    test_set = dataset(
        args.test_list,
        mel_kwargs=feats_cfg,
        use_channels=use_channels
    )

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    teacher = None
    if args.mode in ['kd', 'ad', 'qad']:
        teacher = CRNN(**teacher_cfg).to(device)
        checkpoint = torch.load('./checkpoints/FAC_best.pt')
        teacher.load_state_dict(checkpoint['model_state'], strict=False)
        teacher.eval()
        teacher = teacher.to(device)

    if args.mode in ['qat', 'qad']:
        student = build_qat_student(crnn_cfg, args.pretrained_path)  
        student.train()
        student.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        print("Preparing model for Quantization-Aware Training...")
        student = torch.quantization.prepare_qat(student, inplace=False)
    else:
        student = shuffleFAC(**crnn_cfg).train()
        #load_pretrained_weights(student, args.pretrained_path)

    macs, _ = calculate_macs(student, device, configs)
    total_params, trainable_params = count_parameters(student)

    print("---------------------------------------------------------------")
    print("Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"MACs: {macs}")
    print("---------------------------------------------------------------\n") 


    student = student.to(device)
    
    student_ch_out = crnn_cfg['nb_filters'][-1]
    teacher_ch_out = teacher_cfg['nb_filters'][-1]
    embed_dim = teacher_ch_out
    
    criterion = nn.CrossEntropyLoss()
    if args.mode in ['ad', 'qad']:
        gl_projector = GL_Projector(in_channels=student_ch_out, embed_dim=embed_dim).to(device)
        discriminator = Discriminator(input_dim=embed_dim).to(device)
        discriminator.train()
        params_s = list(student.parameters()) + list(gl_projector.parameters())
        optimizer_s = torch.optim.SGD(params_s, lr=args.lr, weight_decay=1e-4, momentum=0.9, nesterov=False) 
        scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=[args.epoch*0.4,args.epoch*0.8], gamma=0.1) 
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.lr_D, weight_decay=1e-4, momentum=0.9)
        scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[args.epoch*0.4, args.epoch*0.8], gamma=0.1)
    else:
        param = student.parameters()
        optimizer_s = torch.optim.SGD(param, lr=args.lr, weight_decay=1e-4, momentum=0.9, nesterov=False)
        scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=[args.epoch*0.4, args.epoch*0.8], gamma=0.1)


    log_path = './training_log.csv'
    ckpt_dir = './checkpoints'
    exp_dir = './exp'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1'])
    
    best_f1 = -1.0
    num_epochs = args.epoch
    os.makedirs(os.path.join(exp_dir, "devtest_codecarbon"), exist_ok=True)
    tracker = OfflineEmissionsTracker(
        "DCASE Task 4 SED EXP",
        output_dir=os.path.join(exp_dir, "devtest_codecarbon"),
        log_level="warning",
        country_iso_code="KOR",
    )
    tracker.start()
    for epoch in range(1, num_epochs + 1):
        if args.mode in ['ref','qat']:
            train_loss = train(student, train_loader, optimizer_s, criterion, device)
        elif args.mode == 'kd':
            train_loss = kd_train(student, teacher, train_loader, optimizer_s, criterion, device)
        else:
            train_loss = ad_train(student, teacher, gl_projector, discriminator, train_loader, optimizer_s, optimizer_D, epoch, scheduler_D, device)
        val_loss, val_acc, val_f1 = valid(student, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        scheduler_s.step()

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"])

        # If training mode is QAT or QAD, weight model will saving after 10 epoch for stabilizing
        if epoch >= 10 and val_f1 > best_f1 and args.mode in ['qat', 'qad']:
                best_f1 = val_f1
                time_str = datetime.now().strftime("%m%d_%H%M%S")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_s_state_dict': optimizer_s.state_dict(),
                    'scheduler_s_state_dict': scheduler_s.state_dict(),
                    'best_f1': best_f1,
                }
                if args.mode == 'qad':
                    checkpoint['optimizer_D_state_dict'] = optimizer_D.state_dict()
                    checkpoint['gl_projector_state_dict'] = gl_projector.state_dict()
                    checkpoint['discriminator_state_dict'] = discriminator.state_dict()
                    
                print(f"[BEST PERFORMANCE MODEL] : {time_str}\n")
                torch.save(checkpoint, os.path.join(ckpt_dir, f'best_qat_model_{time_str}.pt'))

        elif val_f1 > best_f1 and args.mode in ['kd', 'ad', 'ref']:
            best_f1 = val_f1
            time_str = datetime.now().strftime("%m%d_%H%M%S")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_s_state_dict': optimizer_s.state_dict(),
                'scheduler_s_state_dict': scheduler_s.state_dict(),
                'best_f1': best_f1,
            }
            if args.mode == 'ad':
                checkpoint['optimizer_D_state_dict'] = optimizer_D.state_dict()
                checkpoint['gl_projector_state_dict'] = gl_projector.state_dict()
                checkpoint['discriminator_state_dict'] = discriminator.state_dict()
                
            print(f"[BEST PERFORMANCE MODEL] : {time_str}\n")
            torch.save(checkpoint, os.path.join(ckpt_dir, f'best_model_{time_str}.pt'))
    if args.mode in ['qat', 'qad']:
        print("\nConverting model to INT8...")
        add_safe_globals([scalar, dtype])
        best_qat_model = build_qat_student(crnn_cfg, pretrained_path=None)
        weights_path = os.path.join(ckpt_dir, f'best_qat_model_{time_str}.pt')
        print(f"Load Best Performance Weight : {weights_path}")
        state = torch.load(weights_path, map_location='cpu', weights_only=False)
        best_qat_model.load_state_dict(state['model_state_dict'], strict=False)
        best_qat_model.to('cpu').eval()
        quantized_model = torch.quantization.convert(best_qat_model, inplace=True)

        print("Evaluating final INT8 model...")
        device_cpu = torch.device('cpu')
        test_loss, test_acc, test_macro_f1 = test(quantized_model, test_loader, criterion, device=device_cpu)
        print(f"[FINAL INT8 TEST] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")
        quantized_model.eval()
        exam = torch.randn([1, n_in_channel, 128, 22])
        with torch.no_grad():
            traced = torch.jit.trace(quantized_model, exam.to('cpu'))
        torch.jit.save(traced, f'quantized_classifier_{time_str}.pt')
        print(f"Saved final INT8 model to 'quantized_classifier_{time_str}.pt'")
        emissions = tracker.stop()
        print(f"[CodeCarbon] Estimated emissions: {emissions} kg CO2eq")
    else:
        best_ckpt = os.path.join(ckpt_dir, f'best_model_{time_str}.pt')
        if os.path.exists(best_ckpt):
            state = torch.load(best_ckpt, map_location=device)
            student.load_state_dict(state['model_state_dict'])
            print(f"Loaded best checkpoint (epoch={state.get('epoch')}, best_f1={state.get('best_f1'):.4f})")

        test_loss, test_acc, test_macro_f1 = test(student, test_loader, criterion, device)
        print(f"[TEST] Current Times : {time_str} \n")
        print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")

        emissions = tracker.stop()
        print(f"[CodeCarbon] Estimated emissions: {emissions} kg CO2eq")
    return {
    'test_loss':test_loss,
    'test_acc' : test_acc,
    'test_macro_f1':test_macro_f1}

if __name__ == "__main__":
    results = []
    for i in range(5):
        result = main()
        results.append(result)
    
    import numpy as np
    losses = [r['test_loss'] for r in results]
    accs = [r['test_acc'] for r in results]
    f1s = [r['test_macro_f1'] for r in results]
    
    print("\n" + "="*60)
    print("TOTAL RESULTS (5 experiments)")
    print("="*60)
    print(f"Loss   : {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"Acc    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("="*60)

    sys.exit()
