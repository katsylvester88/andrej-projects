import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters 
batch_size = 32 
block_size = 8 
max_iters = 10000
eval_interval = 300 
learning_rate = 1e-2 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # sad for me 
eval_iters = 200 

# -----------------------
torch.manual_seed(1337) 
# all of shakespeare's works, concatenated 
with open('input.txt', 'r', encoding='utf-8') as f: 
    text = f.read() 
chars = sorted(list(set(text))) 
vocab_size = len(chars) 

# mapping from chars to ints 
# mapping from characters to integers 
# this is tokenizing! 
# Google uses "SentencePiece" https://github.com/google/sentencepiece
# OpenAI uses tiktoken 
# Tradeoff is amount of context vs. size of vocabulary (we have small vocab but lots of context) 
stoi = {ch:i for i,ch in enumerate(chars)} # string to integer 
itos = {i:ch for i,ch in enumerate(chars)} # integer to string 
encode = lambda s: [stoi[c] for c in s] # encodes chars 
decode = lambda l: ''.join([itos[i] for i in l]) # decodes numbers 

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]

def get_batch(split):  
    data = train if split == 'train' else val 
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offsets into training set
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack = stack 1D tensors as rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) 
    return x,y 

# averages loss over multiple batches! less noisy 
@torch.no_grad() # tell pytorch we won't call .backward() on these calcs 
def estimate_loss(): 
    out = {}
    model.eval() # sets model to evaluation phase. (rn useless) 
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters) # hyperparam 
        for k in range(eval_iters): 
            X, Y = get_batch(split) 
            logits, loss = model(X, Y) 
            losses[k] = loss.item() 
        out[split] = losses.mean() 
    model.train() # putting model back in training phase. (rn useless)
    return out 

class BigramLanguageModel(nn.Module): 
    def __init__(self, vocab_size): 
        super().__init__() 
        # rows are current token, columns are possible next token 
        # values of the Embedding (not sure why it's called an Embedding in this case?) 
        # are probabilities 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None): 
        # this will be (B,T,C)
        # B = batch, T = time, C = channel 
        # in our case b is 4, t is 8, c is vocab size or 65 
        # i.e. will work for whole batch at once  
        logits = self.token_embedding_table(idx) 

        if targets is None: 
            loss = None 
        else: 
            # how well are we predicting next char based on logits?  
            # annoyingly, cross_entropy expects (B,C,T)       
            B, T, C = logits.shape 
            logits = logits.view(B*T, C) # stretching array out to be 2D 
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets) 

        return logits, loss
    
    def generate(self, idx, max_new_tokens): 
        # idx = current context of some characters (some batch) 
        # goal: take (B,T) make it (B, T+1), (B, T+2), ... (B, T+max_new_tokens)
        # max_new_tokens = how many more chars we want to generate 
        for _ in range(max_new_tokens): 
            # get predictions
            # apparently calling self will call forward? 
            # that must be the way the nn.Module works
            logits, loss = self(idx) 
            # focus only on last time step (newest char?) 
            logits = logits[:, -1, :] # becomes (B, C) 
            # apply softmax to get probs 
            probs = F.softmax(logits, dim=-1) # still (B,C) 
            # sample from distrn 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) --> 1 for each batch
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx 
    
model = BigramLanguageModel(vocab_size) 
m = model.to(device) 

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters): 
    # every once in a while, evaluate loss on train and val sets
    if iter % eval_interval == 0: 
        losses = estimate_loss() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data 
    xb, yb = get_batch('train')

    # evaluate the loss 
    logits, loss = model(xb, yb) 
    optimizer.zero_grad(set_to_none=True) 
    loss.backward() 
    optimizer.step() 

# generate from the model 
context = torch.zeros((1,1), dtype = torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist())) 
