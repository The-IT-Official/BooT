import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):

  def __init__(self, d_model, d_ffn, dropout=0.1):
    super().__init__()

    hidden_dim = int(2 * d_ffn / 3)

    self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
    self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
    self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):

    gate = F.silu(self.w1(x))
    data = self.w2(x)
    return self.dropout(self.w3(gate * data))

class Embeddings(nn.Module):
  def __init__(self, vocab_size, embed_dim, block_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.pos_enc = nn.Embedding(block_size, embed_dim)

  def forward(self, x):
    B, T = x.size()
    word_embedding = self.embedding(x)
    positions = torch.arange(T, device=x.device)
    pos_embed = self.pos_enc(positions)
    return word_embedding + pos_embed

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, context_size, dropout=0.1):
    super().__init__()

    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
    self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
    self.dropout_p = dropout
    self.resid_dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.qkv(x)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(
      q, k, v, is_causal=True,
      dropout_p=self.dropout_p if self.training else 0.0
    )
    out = out.transpose(1, 2).contiguous().view(B, T, C)

    return self.resid_dropout(self.proj(out))

class Block(nn.Module):
  def __init__(self, n_embed, n_head, block_size):
    super().__init__()
    self.self_attn = MultiHeadAttention(n_embed, n_head, block_size)
    self.ffwd = SwiGLU(n_embed, 4 * n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.self_attn(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BooT(nn.Module):
  def __init__(self, vocab_size, n_embed, n_head, n_layers, block_size):
    super().__init__()

    self.block_size = block_size
    self.embeddings = Embeddings(vocab_size, n_embed, block_size)
    self.blocks = nn.ModuleList([Block(n_embed, n_head, block_size) for _ in range(n_layers)])
    self.lnm = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
    self.lm_head.weight = self.embeddings.embedding.weight

  def forward(self, x, targets=None):
    x = self.embeddings(x)
    for block in self.blocks:
      x = block(x)
    x = self.lnm(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      B, T, V = logits.size()
      loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
    self.eval()
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature

      if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float('-inf')

      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_token), dim=1)
    self.train()
    return idx