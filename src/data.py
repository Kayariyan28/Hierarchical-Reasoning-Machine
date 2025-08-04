import torch
import random
from src.config import VOCAB

def generate_proof_data(num_samples, max_num):
    data = []
    attempts = 0
    min_target = 10
    while len(data) < num_samples and attempts < num_samples * 10:
        attempts += 1
        target = random.randint(min_target, max_num)
        factors = []
        for c in range(1, min(target, max_num // 2)):
            remainder = target - c
            for a in range(2, int(remainder**0.5) + 2):
                if remainder % a == 0:
                    b = remainder // a
                    if b > 1:
                        factors.append((a, b, c))
                        break
            if factors:
                break
        if not factors:
            continue
        a, b, c = random.choice(factors)
        intermediate = a * b
        input_seq = ['[START]', str(target), '[END]']
        output_seq = [
            '[START]', '[LEMMA]', str(a), '*', str(b), '=', str(intermediate),
            '[LEMMA]', str(intermediate), '+', str(c), '=', str(target), '[END]'
        ]
        data.append((input_seq, output_seq))
    return data

def tokenize(seq, max_len):
    tokens = [VOCAB.get(token, VOCAB['[PAD]']) for token in seq]
    tokens += [VOCAB['[PAD]']] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len], dtype=torch.long)

def create_dataloaders(num_samples, max_num, max_len, batch_size, device):
    train_data = generate_proof_data(num_samples, max_num)
    processed_data = []
    for inp, outp in train_data:
        inp_tokens = tokenize(inp, max_len).to(device)
        outp_tokens = tokenize(outp, max_len).to(device)
        processed_data.append((inp_tokens, outp_tokens))

    train_size = int(0.95 * len(processed_data))
    val_size = len(processed_data) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        processed_data, [train_size, val_size], generator=generator
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader