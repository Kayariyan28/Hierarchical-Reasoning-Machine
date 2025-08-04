import torch
import torch.nn as nn
import math

from src.config import device

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:(d_model - 1) // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x

class RecurrentModule(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm_state = nn.LayerNorm(d_model)
        self.norm_ctx1 = nn.LayerNorm(d_model)
        self.norm_ctx2 = nn.LayerNorm(d_model)

    def forward(self, state, context1, context2=None):
        x = self.norm_state(state) + self.norm_ctx1(context1)
        if context2 is not None:
            x = x + self.norm_ctx2(context2)
        for layer in self.layers:
            x = layer(x)
        return x

class HierarchicalReasoningCore(nn.Module):
    def __init__(self, config):
        super(HierarchicalReasoningCore, self).__init__()
        self.config = config
        D_MODEL = config['D_MODEL']
        N_HEADS = config['N_HEADS']
        N_LAYERS = config['N_LAYERS']
        DROPOUT = config['DROPOUT']
        VOCAB_SIZE = config['VOCAB_SIZE']

        self.T = config['T_LOW_LEVEL']
        self.N = config['N_HIGH_LEVEL']
        self.total_internal_steps = self.N * self.T

        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.pos_encoder = PositionalEncoding(D_MODEL, DROPOUT, config['MAX_SEQ_LEN'])
        self.L_module = RecurrentModule(D_MODEL, N_HEADS, N_LAYERS, DROPOUT)
        self.H_module = RecurrentModule(D_MODEL, N_HEADS, N_LAYERS, DROPOUT)
        self.q_head = nn.Linear(D_MODEL, 2)  # [Halt, Continue]

    def forward(self, x_input, z_H_prev, z_L_prev, use_one_step_grad=True):
        x_tilde = self.embedding(x_input) * math.sqrt(self.config['D_MODEL'])
        x_tilde = self.pos_encoder(x_tilde)
        z_H, z_L = z_H_prev, z_L_prev

        steps_no_grad = 0
        if use_one_step_grad and self.training and self.total_internal_steps > 0:
            steps_no_grad = self.total_internal_steps - 1

        if steps_no_grad > 0:
            with torch.no_grad():
                z_H, z_L = self.run_hrm_loop(z_H, z_L, x_tilde, steps_no_grad, start_step=1)

        steps_with_grad = self.total_internal_steps - steps_no_grad
        if steps_with_grad > 0:
            z_H, z_L = self.run_hrm_loop(z_H, z_L, x_tilde, steps_with_grad, start_step=steps_no_grad + 1)
            
        output = self.output_head(z_H)
        q_values = self.q_head(z_H.mean(dim=1))
        return output, z_H, z_L, q_values

    def run_hrm_loop(self, z_H, z_L, x_tilde, steps, start_step=1):
        current_z_H, current_z_L = z_H, z_L
        for i in range(start_step, start_step + steps):
            current_z_L = self.L_module(current_z_L, current_z_H, x_tilde)
            if i % self.T == 0:
                current_z_H = self.H_module(current_z_H, current_z_L)
        return current_z_H, current_z_L

    def initialize_states(self, batch_size, seq_len):
        D_MODEL = self.config['D_MODEL']
        z_H = torch.randn(batch_size, seq_len, D_MODEL, device=device) * 0.02
        z_L = torch.randn(batch_size, seq_len, D_MODEL, device=device) * 0.02
        return z_H, z_L
