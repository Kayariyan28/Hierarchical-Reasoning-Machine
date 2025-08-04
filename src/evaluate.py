import torch
from src.config import VOCAB, INV_VOCAB
from src.data import tokenize

def decode(tokens):
    return " ".join([INV_VOCAB.get(t.item(), '?') for t in tokens if t.item() != VOCAB['[PAD]']])

def simulated_verifier(proof_tokens, input_tokens):
    try:
        target_val_str = INV_VOCAB.get(input_tokens[1].item())
        if not target_val_str.isdigit(): return False, "Invalid Target"
        target_val = int(target_val_str)
    except:
        return False, "Invalid Input"

    proof_str = decode(proof_tokens)
    parts = proof_str.split('[LEMMA]')[1:]
    if len(parts) < 2: return False, "Insufficient Lemmas"

    intermediate_val = -1
    # Verify Lemma 1
    try:
        lemma1 = parts[0].strip().split()
        if '?' in lemma1 or len(lemma1) < 5 or lemma1[1] != '*' or lemma1[3] != '=':
            return False, "Lemma 1 Syntax Error"
        a, b, I = int(lemma1[0]), int(lemma1[2]), int(lemma1[4])
        if a * b == I:
            intermediate_val = I
        else:
            return False, f"Lemma 1 Math Error"
    except:
        return False, "Lemma 1 Parsing Error"

    # Verify Lemma 2
    try:
        lemma2 = parts[1].replace('[END]', '').strip().split()
        if '?' in lemma2 or len(lemma2) < 5 or lemma2[1] != '+' or lemma2[3] != '=':
            return False, "Lemma 2 Syntax Error"
        I_check, c, T = int(lemma2[0]), int(lemma2[2]), int(lemma2[4])
        if I_check != intermediate_val: return False, "Lemma 2 Input Mismatch"
        if I_check + c == T and T == target_val:
            return True, "Verified"
        else:
            return False, "Lemma 2 Math Error"
    except:
        return False, "Lemma 2 Parsing Error"

def evaluate_hlrm(model, input_list, config, device):
    model.eval()
    inp_tokens = tokenize(input_list, config['MAX_SEQ_LEN']).unsqueeze(0).to(device)
    batch_size, seq_len = inp_tokens.shape
    z_H, z_L = model.initialize_states(batch_size, seq_len)
    
    print(f"\n--- HLRM Inference: Proving {' '.join(input_list)} ---")
    verified = False
    M_max = config['MAX_SEGMENTS'] * 2
    
    for m in range(M_max):
        with torch.no_grad():
            output, z_H, z_L, q_values = model(inp_tokens, z_H, z_L, use_one_step_grad=False)
            pred_tokens = torch.argmax(output, dim=-1)
            decoded_output = decode(pred_tokens[0])
            q_halt = q_values[0, 0].item()
            q_continue = q_values[0, 1].item()
            
        print(f"\nSegment {m+1}/{M_max}:")
        print(f" Output: {decoded_output}")
        print(f" Q-Halt: {q_halt:.3f}, Q-Continue: {q_continue:.3f}")
        
        is_verified, reason = simulated_verifier(pred_tokens[0], inp_tokens[0])
        if is_verified:
            print(f" Status: SUCCESS ({reason})")
            verified = True
            break
        
        if m >= 0 and q_halt > q_continue + 0.05:
            print(f" Status: ACT Halted. (Verifier Reason: {reason})")
            break
            
        print(f" Status: Refining (Verifier Reason: {reason})...")
        
    if not verified:
        print("\nResult: Failed to prove within budget.")
    return verified