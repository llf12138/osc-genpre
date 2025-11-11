import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import time
import os

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


RDLogger.DisableLog('rdApp.*')

class Generator:
    def __init__(self, model_path, vocab_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load vocabulary
        vocab = torch.load(vocab_path, map_location=self.device)
        self.char_to_idx = vocab['char_to_idx']
        self.idx_to_char = vocab['idx_to_char']
        self.vocab_size = len(self.char_to_idx)

        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_batch(self, batch_size, max_length):

        seed_idx = self.char_to_idx['?']
        input_seq = torch.full((batch_size, 1), seed_idx, device=self.device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        sequences = [['?'] for _ in range(batch_size)]

        for _ in range(max_length):
            with torch.no_grad():
                output = self.model(input_seq)
            output = output[:, -1, :]
            probabilities = torch.softmax(output, dim=-1)
            
            next_idxs = torch.multinomial(probabilities, 1).view(-1)

            for i in range(batch_size):
                if not finished[i]:
                    char = self.idx_to_char[next_idxs[i].item()]
                    sequences[i].append(char)
                    if char == '+':
                        finished[i] = True
            
            if finished.all():
                break
            
            input_seq = torch.cat([input_seq, next_idxs.unsqueeze(1)], dim=1)
        return [''.join(seq).replace('?', '').replace('+', '') for seq in sequences]

def is_valid_smiles(smiles):
    if not smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    model_path = "model_LSTM.pt"
    vocab_path = "vocab.pt"
    output_file = "generated_smiles.txt"
    valid_smiles_file = "valid_smiles.txt"
    data_file = "data.txt"
    batch_size = 512
    num_batches = 100
    max_length = 295

    print("Loading generator model...")
    start_time = time.time()
    generator = Generator(model_path=model_path, vocab_path=vocab_path)
    print(f"Model loaded in {time.time()-start_time:.2f} seconds")

    existing_smiles = set()
    if os.path.exists(data_file):
        print(f"Loading existing SMILES from {data_file}...")
        with open(data_file, 'r') as f:
            for line in f:
                existing_smiles.add(line.strip())
        print(f"Loaded {len(existing_smiles)} existing SMILES")
    else:
        print(f"No existing data file found at {data_file}")

    with open(output_file, 'w') as out_file, open(valid_smiles_file, 'w') as valid_file:
        total_generated = 0
        total_valid_unique = 0
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            print(f"\nGenerating batch {batch_idx+1}/{num_batches}...")

            smiles_batch = generator.generate_batch(
                batch_size=batch_size,
                max_length=max_length
            )
            
            for smi in smiles_batch:
                out_file.write(smi + '\n')
            total_generated += len(smiles_batch)
            
            valid_unique = set()
            for smi in smiles_batch:
                if smi and smi not in existing_smiles and is_valid_smiles(smi):
                    valid_unique.add(smi)
            
            for smi in valid_unique:
                valid_file.write(smi + '\n')
                existing_smiles.add(smi)
            
            num_valid = len(valid_unique)
            total_valid_unique += num_valid
            batch_time = time.time() - batch_start
            
            print(f"Generated: {len(smiles_batch)}, Valid unique: {num_valid}")
            print(f"Batch time: {batch_time:.2f}s, "
                  f"Speed: {len(smiles_batch)/batch_time:.1f} SMILES/s")
            print(f"Validity rate: {num_valid/len(smiles_batch)*100:.1f}%")

    print("\n" + "="*50)
    print(f"Total generated SMILES: {total_generated}")
    print(f"Total valid unique SMILES: {total_valid_unique}")
    print(f"Overall validity rate: {total_valid_unique/total_generated*100:.1f}%")
    print(f"Results saved to {output_file} and {valid_smiles_file}")

if __name__ == "__main__":
    main()