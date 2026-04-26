import urllib.request
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Output paths (resolved relative to this script, not the shell cwd) ──────
HERE     = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(HERE, 'MLP_ass4', 'Figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(HERE, 'sherlock.txt')
SEQ_LENGTH       = 100
HIDDEN_DIM       = 700
LAYER_NUM        = 3
BATCH_SIZE       = 12
NUM_EPOCHS       = 20
TEST_SPLIT       = 0.10
GENERATE_LENGTH  = 20
CHECKPOINT_EVERY = 10
RESULTS_FILE     = os.path.join(HERE, 'results.json')

# ── Download dataset ─────────────────────────────────────────────────────────
if not os.path.exists(DATA_DIR):
    print('Downloading Sherlock Holmes corpus...')
    urllib.request.urlretrieve(
        'https://www.gutenberg.org/cache/epub/1661/pg1661.txt',
        DATA_DIR
    )
print(f'Dataset ready: {DATA_DIR} ({os.path.getsize(DATA_DIR):,} bytes)')

# ── Preprocessing and vocabulary ─────────────────────────────────────────────
data             = open(DATA_DIR, 'r', encoding='latin-1').read()
valid_characters = string.ascii_letters + ".,! -'" + string.digits
character_to_int = {c: i for i, c in enumerate(valid_characters)}
int_to_character = {i: c for i, c in enumerate(valid_characters)}

training_string = ''
for character in data:
    if character in valid_characters:
        training_string += character
    elif character == '\n':
        training_string += ' '
while '  ' in training_string:
    training_string = training_string.replace('  ', ' ')

target_string   = training_string[1:]
training_string = training_string[:-1]

print(f'Cleaned corpus : {len(training_string):,} characters')
print(f'Vocabulary size: {len(valid_characters)} symbols')

# ── Build sequences ──────────────────────────────────────────────────────────
X_list, y_list = [], []
for i in range(0, len(training_string), SEQ_LENGTH):
    training_sequence = training_string[i : i + SEQ_LENGTH]
    int_train_seq     = [character_to_int[v] for v in training_sequence]
    input_sequence    = np.zeros((SEQ_LENGTH, len(valid_characters)), dtype='float32')
    if len(int_train_seq) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            input_sequence[j][int_train_seq[j]] = 1.
    X_list.append(input_sequence)

    y_sequence = target_string[i : i + SEQ_LENGTH]
    y_seq_ix   = [character_to_int[v] for v in y_sequence]
    target_seq = np.zeros((SEQ_LENGTH, len(valid_characters)), dtype='float32')
    if len(y_seq_ix) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            target_seq[j][y_seq_ix[j]] = 1.
    y_list.append(target_seq)

X = np.reshape(X_list, (-1, SEQ_LENGTH, len(valid_characters)))
y = np.reshape(y_list, (-1, SEQ_LENGTH, len(valid_characters)))

# ── Train / test split (sequential) ─────────────────────────────────────────
split           = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'Total sequences : {len(X):,}')
print(f'Training        : {len(X_train):,}')
print(f'Test (held-out) : {len(X_test):,}')

# ── DataLoaders ──────────────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=BATCH_SIZE, shuffle=False
)
print(f'Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}')

# ── Model ────────────────────────────────────────────────────────────────────
class TextGeneratorLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_dim, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = TextGeneratorLSTM(
                len(valid_characters), HIDDEN_DIM, LAYER_NUM,
                len(valid_characters)
            ).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

n_params = sum(p.numel() for p in model.parameters())
print(f'Device     : {device}')
print(f'Parameters : {n_params:,}')

# ── Text generation ──────────────────────────────────────────────────────────
def generate_text(model, length=GENERATE_LENGTH, seed=42):
    model.eval()
    np.random.seed(seed)
    ix    = [np.random.randint(len(valid_characters))]
    X_gen = np.zeros((1, length, len(valid_characters)), dtype='float32')
    chars = []
    with torch.no_grad():
        for i in range(length):
            X_gen[0, i, ix[-1]] = 1
            out = model(torch.tensor(X_gen[:, :i + 1, :]).to(device))
            ix  = [out[0, -1, :].argmax().item()]
            chars.append(int_to_character[ix[-1]])
    return ''.join(chars)

# ── Training loop ────────────────────────────────────────────────────────────
train_losses, test_losses, samples = [], [], []

print(f'\n{"Epoch":>6}  {"Train Loss":>10}  {"Test Loss":>9}  Sample (20 chars)')
print('-' * 70)

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output      = model(X_batch)
        output_flat = output.view(-1, len(valid_characters))
        y_flat      = y_batch.view(-1, len(valid_characters)).argmax(dim=1)
        loss        = criterion(output_flat, y_flat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)

    model.eval()
    test_loss_acc = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output      = model(X_batch)
            output_flat = output.view(-1, len(valid_characters))
            y_flat      = y_batch.view(-1, len(valid_characters)).argmax(dim=1)
            test_loss_acc += criterion(output_flat, y_flat).item()
    test_loss = test_loss_acc / len(test_loader)

    sample = generate_text(model, GENERATE_LENGTH, seed=42)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    samples.append({
        'epoch':      epoch,
        'train_loss': round(train_loss, 6),
        'test_loss':  round(test_loss,  6),
        'sample':     sample
    })

    print(f'{epoch:>6}  {train_loss:>10.4f}  {test_loss:>9.4f}  "{sample}"')

    # Flush to disk after every epoch — survives cluster job interruption
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'vocab_size':  len(valid_characters),
            'seq_length':  SEQ_LENGTH,
            'hidden_dim':  HIDDEN_DIM,
            'num_layers':  LAYER_NUM,
            'batch_size':  BATCH_SIZE,
            'n_params':    n_params,
            'train_seqs':  len(X_train),
            'test_seqs':   len(X_test),
            'epochs':      samples
        }, f, indent=2)

    if epoch % CHECKPOINT_EVERY == 0:
        ckpt = os.path.join(HERE, f'checkpoint_{HIDDEN_DIM}_epoch_{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f'         Checkpoint saved → {ckpt}')

# ── Loss figure ──────────────────────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs_range, train_losses, color='steelblue',  linewidth=1.8,
        marker='o', markersize=4, label='Training loss')
ax.plot(epochs_range, test_losses,  color='darkorange', linewidth=1.8,
        marker='s', markersize=4, linestyle='--', label='Test loss (held-out 10 %)')
ax.axhline(math.log(len(valid_characters)), color='tomato', linestyle=':',
           linewidth=1.0,
           label=f'Random baseline ln({len(valid_characters)})={math.log(len(valid_characters)):.2f}')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax.set_title('Training vs. Test Loss — 3-Layer LSTM on Sherlock Holmes', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.4, alpha=0.5)
plt.tight_layout()

for path in [
    os.path.join(FIGS_DIR, 'training_loss.pdf'),
    os.path.join(FIGS_DIR, 'training_loss.png'),
    os.path.join(HERE,     'training_loss.pdf'),
    os.path.join(HERE,     'training_loss.png'),
]:
    plt.savefig(path, bbox_inches='tight', dpi=150)

# ── Final 20-character generation (Assignment Objective 2) ───────────────────
final_sample = generate_text(model, GENERATE_LENGTH, seed=42)
print('\n' + '━' * 42)
print(f'Final generated sequence ({GENERATE_LENGTH} characters):')
print(f'  "{final_sample}"')
print('━' * 42)

print('\nFiles written:')
print(f'  {RESULTS_FILE}')
print(f'  {os.path.join(FIGS_DIR, "training_loss.pdf")}')
print(f'  {os.path.join(FIGS_DIR, "training_loss.png")}')
print(f'  {os.path.join(HERE, "training_loss.pdf")}  (fallback)')
