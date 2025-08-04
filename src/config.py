import torch

# --- Global Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'D_MODEL': 128, 'N_HEADS': 8, 'N_LAYERS': 2,
    'T_LOW_LEVEL': 3, 'N_HIGH_LEVEL': 2,
    'MAX_SEGMENTS': 4, 'MAX_SEQ_LEN': 20,
    'LR': 0.0005, 'BATCH_SIZE': 64, 'EPOCHS': 200,
    'MAX_NUMBER': 50, 'DROPOUT': 0.1,
}

# --- Vocabulary ---
MAX_NUM = CONFIG['MAX_NUMBER']
VOCAB = {str(i): i for i in range(MAX_NUM + 1)}
SPECIAL_TOKENS = ['+', '*', '=', '[START]', '[END]', '[PAD]', '[LEMMA]']
for token in SPECIAL_TOKENS:
    VOCAB[token] = len(VOCAB)

INV_VOCAB = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

# Add VOCAB_SIZE to config so the model can access it during init
CONFIG['VOCAB_SIZE'] = VOCAB_SIZE