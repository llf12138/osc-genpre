import torch
import random
import pandas as pd

random.seed(42)
torch.manual_seed(42)

data = pd.read_csv("data_A_gen.csv")
smiles_list = data['smiles'].tolist()

special_tokens = ["@", "+"]
atom_symbols = set()

for smiles in smiles_list:
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":
            j = smiles.find("]", i)
            if j != -1:
                atom_symbols.add(smiles[i : j + 1])
                i = j + 1
            else:
                raise ValueError(f"Unmatched bracket: {smiles}")
        elif i + 1 < len(smiles) and smiles[i : i + 2] in ["Cl", "Br"]:
            atom_symbols.add(smiles[i : i + 2])
            i += 2
        else:
            atom_symbols.add(smiles[i])
            i += 1

vocab = sorted(list(atom_symbols)) + special_tokens
vocab_dict = {char: idx for idx, char in enumerate(vocab)}
reverse_vocab_dict = {idx: char for char, idx in vocab_dict.items()}

vocab_complete = {"char_to_idx": vocab_dict, "idx_to_char": reverse_vocab_dict}
torch.save(vocab_complete, "vocab.pt")

max_len = max(len(smiles) for smiles in smiles_list) + 1

def encode_smiles(smiles, vocab_dict, max_len):
    encoded = [vocab_dict["@"]]
    i = 0
    while i < len(smiles):
        if smiles[i] == "[":
            j = smiles.find("]", i)
            if j != -1:
                token = smiles[i : j + 1]
                if token not in vocab_dict:
                    raise ValueError(f"Unknown token: {token}")
                encoded.append(vocab_dict[token])
                i = j + 1
            else:
                raise ValueError(f"Unmatched bracket: {smiles}")
        elif i + 1 < len(smiles) and smiles[i : i + 2] in vocab_dict:
            token = smiles[i : i + 2]
            encoded.append(vocab_dict[token])
            i += 2
        else:
            token = smiles[i]
            if token not in vocab_dict:
                raise ValueError(f"Unknown token: {token}")
            encoded.append(vocab_dict[token])
            i += 1
    while len(encoded) < max_len:
        encoded.append(vocab_dict["+"])
    return encoded

encoded_smiles = [encode_smiles(smiles, vocab_dict, max_len) for smiles in smiles_list]

smiles_tensor = torch.tensor(encoded_smiles, dtype=torch.long)

total_samples = len(smiles_tensor)
indices = list(range(total_samples))
random.shuffle(indices)

train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size :]

train_tensor = smiles_tensor[train_indices]
val_tensor = smiles_tensor[val_indices]
test_tensor = smiles_tensor[test_indices]

torch.save({
    "train": train_tensor,
    "val": val_tensor,
    "test": test_tensor
}, "smiles.pt")

print("Vocabulary content:")
print(vocab_complete)
print("\nTraining set size:", train_tensor.size(0))
print("Validation set size:", val_tensor.size(0))
print("Test set size:", test_tensor.size(0))
print("\nLast molecule encoding (test set):")
print(test_tensor[-1].tolist())