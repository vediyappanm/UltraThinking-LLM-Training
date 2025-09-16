# min.py - 100 line mini GPT with temp + top-k
import torch, torch.nn as nn, torch.nn.functional as F

# ------------------- Config -------------------
batch_size = 32
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd, n_head, n_layer, dropout = 128, 4, 2, 0.1
temperature, top_k = 0.8, 50

# ------------------- Data ---------------------
text = open("input.txt", "r", encoding="utf-8").read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split=="train" else val_data
    ix = torch.randint(len(d)-block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ------------------- Model --------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2,-1) / C**0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd); self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead(n_head, n_embd//n_head)
        self.ff = nn.Sequential(nn.Linear(n_embd,n_embd*4), nn.ReLU(), nn.Linear(n_embd*4,n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self,x): x = x + self.sa(self.ln1(x)); return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(len(chars), n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, len(chars))
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok, pos = self.tok_emb(idx), self.pos_emb(torch.arange(T, device=device))
        x = tok + pos; x = self.blocks(x); x = self.ln_f(x); logits = self.head(x)
        if targets is None: return logits, None
        return logits, F.cross_entropy(logits.view(-1, len(chars)), targets.view(-1))
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:] / temperature
            probs = F.softmax(logits, dim=-1)
            if top_k:
                v,i = torch.topk(probs, min(top_k, probs.size(-1)))
                probs = torch.zeros_like(probs).scatter_(1,i,v); probs /= probs.sum()
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPT().to(device); opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ------------------- Train --------------------
for step in range(max_iters+1):
    xb,yb = get_batch("train")
    logits,loss = model(xb,yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % eval_interval==0:
        print(f"step {step} | loss {loss.item():.4f}")

# ------------------- Sample ------------------
context = torch.zeros((1,1), dtype=torch.long, device=device)
print("\n--- SAMPLE ---\n"+decode(model.generate(context, 200)[0].tolist())+"\n--------------")
