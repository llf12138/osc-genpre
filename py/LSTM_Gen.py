import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import time
import math

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hid_size, n_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hid_size, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out)

class GEN(nn.Module):
    def __init__(self, emb_size, dic_size, hid_size, n_layers, emb_dropout, dropout):
        super(GEN, self).__init__()
        self.emb = nn.Embedding(dic_size, emb_size, padding_idx=0)
        self.drop = nn.Dropout(emb_dropout)
        self.encoder = LSTMEncoder(emb_size, hid_size, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hid_size, dic_size)

    def forward(self, input):
        emb = self.drop(self.emb(input))
        y = self.encoder(emb)
        o = self.decoder(y)
        return o.contiguous()

def evaluate(model, data_iter, criterion, vocab_size, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, _ in data_iter:
            targets = data[:, 1:].to(device)
            inputs = data[:, :-1].to(device)

            outputs = model(inputs)

            final_output = outputs.contiguous().view(-1, vocab_size)
            final_target = targets.contiguous().view(-1)

            loss = criterion(final_output, final_target)
            total_loss += loss.item()

    return total_loss / len(data_iter)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 32
    dropout = 0.1
    emb_dropout = 0.2
    epochs = 100
    emb_size = 64
    n_layers = 3
    lr = 0.001
    hid_size = 128

    torch.manual_seed(1024)

    vocab_path = "vocab.pt"
    vocab = torch.load(vocab_path)
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    split_data = torch.load("smiles.pt")
    train_tensor = split_data["train"]
    val_tensor = split_data["val"]
    test_tensor = split_data["test"]

    train_inputs = train_tensor[:, :-1]
    train_targets = train_tensor[:, 1:]
    train_dataset = TensorDataset(train_inputs, train_targets)

    val_inputs = val_tensor[:, :-1]
    val_targets = val_tensor[:, 1:]
    val_dataset = TensorDataset(val_inputs, val_targets)

    test_inputs = test_tensor[:, :-1]
    test_targets = test_tensor[:, 1:]
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data loaded successfully.")

    model = GEN(emb_size=emb_size, dic_size=vocab_size, hid_size=hid_size, n_layers=n_layers,
                emb_dropout=emb_dropout, dropout=dropout)
    model = model.to(device)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    best_val_loss = float('inf')

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            start_time = time.time()

            for data, _ in train_loader:
                targets = data[:, 1:].to(device)
                inputs = data[:, :-1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                final_output = outputs.contiguous().view(-1, vocab_size)
                final_target = targets.contiguous().view(-1)

                loss = criterion(final_output, final_target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time
            print(f'| Epoch: {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Time: {epoch_time:.2f}s')

            val_loss = evaluate(model, val_loader, criterion, vocab_size, device)
            scheduler.step(val_loss)
            print(f'| Val Loss: {val_loss:.6f} | Val Perplexity: {math.exp(val_loss):.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model_LSTM.pt")
                torch.save(model, "model_LSTM.pt")
                print(f'| Save model | (Val Loss: {val_loss:.6f})')

            print('-' * 50)

    except KeyboardInterrupt:
        print('Training interrupted by user.')

    model.load_state_dict(torch.load("best_model_LSTM.pt"))
    model.eval()
    test_loss = evaluate(model, test_loader, criterion, vocab_size, device)
    print('=' * 50)
    print(f'| Test Loss: {test_loss:.4f} | Test Perplexity: {math.exp(test_loss):.4f}')
    print('=' * 50)

if __name__ == "__main__":
    main()