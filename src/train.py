import torch
import torch.nn as nn
from src.config import VOCAB

mse_loss = nn.MSELoss()

def calculate_accuracy(preds, targets):
    pred_tokens = torch.argmax(preds, dim=-1)
    correct_tokens = (pred_tokens == targets).float()
    pad_mask = (targets != VOCAB['[PAD]']).float()
    if pad_mask.sum() == 0:
        return 0.0, torch.zeros(targets.shape[0], device=targets.device)
    seq_correct = ((correct_tokens * pad_mask).sum(dim=1) == pad_mask.sum(dim=1)).float()
    seq_acc = seq_correct.mean().item()
    return seq_acc, seq_correct

def train_step(model, dataloader, optimizer, criterion, scheduler, config):
    model.train()
    total_loss, total_seq_acc = 0, 0
    for inputs, targets in dataloader:
        batch_size, seq_len = inputs.shape
        optimizer.zero_grad()

        z_H, z_L = model.initialize_states(batch_size, seq_len)
        segment_losses, q_values_list, predictions = [], [], []

        for m in range(config['MAX_SEGMENTS']):
            z_H, z_L = z_H.detach(), z_L.detach()
            output, z_H, z_L, q_values = model(inputs, z_H, z_L, use_one_step_grad=True)
            loss = criterion(output.view(-1, config['VOCAB_SIZE']), targets.view(-1))
            segment_losses.append(loss)
            q_values_list.append(q_values)
            predictions.append(output)

        seq_loss = sum(segment_losses) / len(segment_losses)
        _, seq_correct_mask = calculate_accuracy(predictions[-1], targets)
        rewards = seq_correct_mask
        act_loss = 0
        R = rewards
        for m in reversed(range(config['MAX_SEGMENTS'])):
            Q_vals = q_values_list[m]
            Target_Q = torch.zeros_like(Q_vals)
            Target_Q[:, 0] = R  # Halt target
            if m < config['MAX_SEGMENTS'] - 1:
                Target_Q[:, 1] = R  # Continue target
            act_loss += mse_loss(Q_vals, Target_Q.detach())

        loss = seq_loss + 0.05 * act_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        seq_acc, _ = calculate_accuracy(predictions[-1], targets)
        total_loss += loss.item()
        total_seq_acc += seq_acc
    
    scheduler.step()
    N = len(dataloader)
    return total_loss / N, total_seq_acc / N

def validate(model, dataloader, criterion):
    model.eval()
    total_seq_acc = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            batch_size, seq_len = inputs.shape
            z_H, z_L = model.initialize_states(batch_size, seq_len)
            output, _, _, _ = model(inputs, z_H, z_L, use_one_step_grad=False)
            seq_acc, _ = calculate_accuracy(output, targets)
            total_seq_acc += seq_acc
    return total_seq_acc / len(dataloader)