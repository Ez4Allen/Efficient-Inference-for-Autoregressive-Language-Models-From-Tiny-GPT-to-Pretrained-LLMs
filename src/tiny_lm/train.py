import os
import torch
from torch.utils.data import DataLoader

from src.tiny_lm.dataset import (
    TextDataset,
    create_train_val_split,
    load_text_file,
)
from src.tiny_lm.tokenizer import CharTokenizer
from src.tiny_lm.model import TinyGPT


def evaluate(model, data_loader, device, max_batches=20):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    # -------------------------
    # Basic config
    # -------------------------
    data_path = "data/tiny_shakespeare/input.txt"
    output_dir = "results/tiny_gpt_shakespeare"

    block_size = 128
    batch_size = 32
    n_layer = 2
    n_head = 4
    n_embd = 128
    dropout = 0.1

    learning_rate = 3e-4
    num_epochs = 5
    eval_every = 200

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------
    # Load text and tokenizer
    # -------------------------
    text = load_text_file(data_path)

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    token_ids = tokenizer.encode(text)

    train_ids, val_ids = create_train_val_split(token_ids, val_ratio=0.1)

    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    print("Vocab size:", tokenizer.vocab_size)
    print("Train tokens:", len(train_ids))
    print("Val tokens:", len(val_ids))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # -------------------------
    # Model
    # -------------------------
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters:", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # -------------------------
    # Training loop
    # -------------------------
    global_step = 0

    model.train()

    for epoch in range(num_epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)

                print(
                    f"epoch {epoch+1}/{num_epochs} | "
                    f"step {global_step} | "
                    f"train loss {loss.item():.4f} | "
                    f"val loss {val_loss:.4f}"
                )

            global_step += 1

    # -------------------------
    # Save checkpoint + tokenizer
    # -------------------------
    ckpt_path = os.path.join(output_dir, "model.pt")
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "block_size": block_size,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "dropout": dropout,
            },
        },
        ckpt_path,
    )

    tokenizer.save(tokenizer_path)

    print("Saved model to:", ckpt_path)
    print("Saved tokenizer to:", tokenizer_path)


if __name__ == "__main__":
    main()