import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Dataset for GPT-style autoregressive language modeling.

    Given a long sequence of token ids, each sample returns:
        x = tokens[i : i + block_size]
        y = tokens[i + 1 : i + block_size + 1]

    The model learns to predict y from x, i.e. next-token prediction.
    """

    def __init__(self, token_ids, block_size):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D sequence of token ids.")

        if len(token_ids) <= block_size:
            raise ValueError("token_ids length must be greater than block_size.")

        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_train_val_split(token_ids, val_ratio=0.1):
    """
    Split token ids into train and validation parts.

    For language modeling, we usually split the long token stream
    contiguously instead of randomly shuffling individual tokens.
    """

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    n = len(token_ids)
    val_size = int(n * val_ratio)
    train_size = n - val_size

    train_ids = token_ids[:train_size]
    val_ids = token_ids[train_size:]

    return train_ids, val_ids


def load_text_file(path):
    """
    Load a raw text file as a single string.
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return text