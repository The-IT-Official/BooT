import os 
import math
import torch
import tiktoken
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm 
from model import BooT

batch_size = 64
lr = 3e-4
n_embed = 768
device = ("cuda"
          if torch.cuda.is_available()
          else "cpu")
n_layers = 12
n_head = 12
block_size = 1024
epochs = 15000
warmup_steps = 300
target_tokens = 200_000_000
save_dir = "./checkpoints"

torch.manual_seed(42)
os.makedirs(save_dir, exist_ok=True)

enc = tiktoken.get_encoding('gpt2')
encoder = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decoder = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

print("Streaming FineWeb-Edu dataset...")

dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

one_dim_tokens = []

for item in dataset:
    tokens = encoder(item["text"])
    one_dim_tokens.extend(tokens)

    if len(one_dim_tokens) >= target_tokens:
      break

data = torch.tensor(one_dim_tokens, dtype=torch.long)
# train/test split
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

print(f"Train tokens: {len(train_data):,} | Val tokens: {len(test_data):,}")

def get_batches(split):
  data = train_data if split == 'train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

model = BooT(vocab_size, n_embed, n_head, n_layers, block_size)
model.to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

model = torch.compile(model)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, epochs - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))

sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

@torch.no_grad()
def estimate_val_loss(batches=20):
  model.eval()
  losses = []
  for _ in range(batches):
    x, y = get_batches("test")
    _, loss = model(x, y)
    losses.append(loss.item())
  model.train()
  return sum(losses) / len(losses)

train_losses, val_losses, val_steps = [], [], []
pbar = tqdm(range(epochs))

for epoch in pbar:
    x, y = get_batches("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step()
    sched.step()

    train_losses.append(loss.item())
    pbar.set_description(f"loss {loss.item():.4f}")

    if epoch % 500 == 0 and epoch > 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "epoch": epoch,
        }, f"/content/drive/MyDrive/checkpoint_{epoch}.pt")  # Drive, not /content/

    if epoch % 200 == 0:
        val_loss = estimate_val_loss(batches=10)
        val_losses.append(val_loss)
        val_steps.append(epoch)
        pbar.write(f"epoch {epoch}, train loss {loss.item():.4f}, val loss {val_loss:.4f}")

final_val_loss = estimate_val_loss(batches=20)
perplexity = math.exp(final_val_loss)
print(f"Final val loss: {final_val_loss:.4f} | Perplexity: {perplexity:.2f}")

plt.plot(range(len(train_losses)), train_losses, label="train", alpha=0.5)
plt.plot(val_steps, val_losses, label="val", marker='o')
plt.xlabel("step") 
plt.ylabel("loss")
plt.legend(); 
plt.title("BooT_GPT training curve")
plt.savefig("/content/drive/MyDrive/loss_curve.png")