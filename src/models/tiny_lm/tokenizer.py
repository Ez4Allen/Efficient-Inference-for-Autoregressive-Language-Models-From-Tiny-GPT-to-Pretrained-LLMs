import json

class CharTokenizer:
    """
    Simple character-level tokenizer for tiny GPT-style language modeling.

    It maps each unique character in the training text to an integer token id.
    This is enough for Tiny Shakespeare and useful for debugging the full LM pipeline.
    """

    def __init__(self):
        self.stoi = None  # string-to-index
        self.itos = None  # index-to-string
        self.vocab_size = None

    def fit(self, text):
        """
        Build vocabulary from raw text.
        """
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        """
        Convert text string into a list of token ids.
        """
        if self.stoi is None:
            raise ValueError("Tokenizer has not been fitted yet.")

        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """
        Convert a list of token ids back into a text string.
        """
        if self.itos is None:
            raise ValueError("Tokenizer has not been fitted yet.")

        return "".join([self.itos[int(i)] for i in ids])

    def save(self, path):
        """
        Save tokenizer vocabulary to a JSON file.
        """
        if self.stoi is None or self.itos is None:
            raise ValueError("Tokenizer has not been fitted yet.")

        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """
        Load tokenizer vocabulary from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.stoi = data["stoi"]
        tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        tokenizer.vocab_size = data["vocab_size"]

        return tokenizer