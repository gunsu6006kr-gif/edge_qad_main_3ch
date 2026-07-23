import torch
import torch.nn as nn
from .losses import apply_specaugment, kd_loss_logits, loss_proj2, loss_mvg, loss_MVD
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F

def train(student, train_loader, optimizer, criterion, device):
    student.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        student_outputs = student(batch_x)
        loss_student = criterion(student_outputs, batch_y)

        loss_student.backward()
        optimizer.step()
        total_loss += loss_student.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)

def kd_train(student, teacher, train_loader, optimizer, criterion, device):
    student.train()
    teacher.eval()
    train_loss = 0.0
    num_batches = 0
    alpha = 0.35
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_outputs = teacher(batch_x)
        student_outputs = student(batch_x)
        loss_student = criterion(student_outputs, batch_y)

        loss_distill = kd_loss_logits(student_outputs, teacher_outputs, T=4)

        total_loss = alpha * loss_distill + (1 - alpha) * loss_student
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        num_batches += 1
        
    return train_loss / max(1, num_batches)

def ad_train(student, teacher, gl_projector, discriminator,train_loader, optimizer_s,optimizer_D, epoch, scheduler_D, device):
    student.train()
    teacher.eval()
    discriminator.train()

    criterion_cls = nn.CrossEntropyLoss()
    train_loss = 0.0
    num_batches = 0
    d_ratio = 5
    lam = 1e-4
    for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True)):
        batch_x = batch_x.to(device, non_blocking=True) # ([Batch, 1, 128, 22]) ([B, C, F, T])
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer_s.zero_grad(set_to_none=True)
        x_aug = apply_specaugment(batch_x=batch_x).to(device)
        student_logits, h_s = student(x_aug, return_feats=True)
        with torch.no_grad():
            teacher_logits, h_T = teacher(batch_x, return_feats=True)
        loss_kd = kd_loss_logits(student_logits, teacher_logits, T=4)
        proj_student = gl_projector(h_s)
        loss_cls = criterion_cls(student_logits, batch_y)
        loss_align = loss_proj2(h_T.detach(), proj_student)
        if epoch > 10:
            loss_adv_student = loss_mvg(discriminator, proj_student)
        else:
            loss_adv_student = torch.tensor(0.0).to(device)
        loss_total = loss_cls + 0.35*loss_kd + 0.1*loss_align + lam * loss_adv_student
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(gl_projector.parameters()), 5.0)
        optimizer_s.step()

        if i % d_ratio == 0 and epoch > 10:
            optimizer_D.zero_grad()
            loss_adv_disc = loss_MVD(discriminator, h_T.detach(), proj_student.detach())
            loss_adv_disc.backward()
            optimizer_D.step()
            scheduler_D.step()

        train_loss += loss_total.item()
        num_batches += 1

    return train_loss / max(1, num_batches)

@torch.no_grad()
def valid(student, val_loader, criterion, device):
    student.eval()
    val_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(val_loader, total=len(val_loader), desc='Valid', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        num_batches += 1
        pred = torch.argmax(outputs, dim=1)
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if len(y_true_all) == 0:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    val_acc = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average='macro')

    return (val_loss / max(1, num_batches)), val_acc, val_macro_f1

@torch.no_grad()
def test(student, test_loader, criterion, device):
    student.eval()
    test_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(test_loader, total=len(test_loader), desc='Test', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        num_batches += 1
        pred = torch.argmax(outputs, dim=1)
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if len(y_true_all) == 0:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return (test_loss / max(1, num_batches)), test_acc, test_macro_f1