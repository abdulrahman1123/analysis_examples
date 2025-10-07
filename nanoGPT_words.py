'''
This code is modified from https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing,
which is explained in this video https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2409s
I modified it to take words as tokens
'''
### Import libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

torch.manual_seed(111)

# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel... will be referred to as B in the following code
block_size = 256 # what is the maximum context length for predictions ... will be referred to as T in the following code

max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 6
n_layer = 6
n_embd = (n_embd//n_head)*n_head # inside the code, head_size is calculated as n_embd//n_head, which might give an error if the result are not a full integer
dropout = 0.2
# ------------

# Load the required data. Here we use tinyShakespeare. You can use your own data if you want.
url1 = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
url1='https://www.gutenberg.org/cache/epub/100/pg100.txt'
url2 = 'https://www.gutenberg.org/cache/epub/11/pg11.txt' # Alice in wonderland
url3 = 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt' # pride and prejudice
url4 = 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt' # moby dick
url5 = 'https://www.gutenberg.org/cache/epub/84/pg84.txt'
url6 = 'https://www.gutenberg.org/cache/epub/145/pg145.txt'
url7 = 'https://www.gutenberg.org/cache/epub/67979/pg67979.txt'
texts = [requests.get(url).text for url in [url1,url2,url3,url4,url5,url6,url7]] # gt the contents of the url. This gives a response that includes response.text (among other things)
texts = [text[0:text.find('*** END OF THE PROJECT GUTENBERG')].replace('\r','').replace('\n\n','\r\r').replace('\n',' ').replace('\r\r','\n') for text in texts]
text = '\n\n'.join(texts)

# check the downloaded text
print("length of dataset in characters: ", len(text),'\n ------------------------------------')
print(text[:500])



# Initialize a ByteLevel BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
trainer = trainers.BpeTrainer(
    vocab_size=6000,
    min_frequency=2,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

# ByteLevel pre-tokenizer (splits into byte-level pieces, can fully reconstruct text)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# Train directly on your dataset string (in-memory, no saving to disk)
tokenizer.train_from_iterator([text], trainer=trainer)

# ---------------------------
# Example usage
# ---------------------------

s = "The quick brown fox jumps over the lazy dog"

encoded = tokenizer.encode(s)
print("Encoded IDs:", encoded.ids)
print("Tokens:", encoded.tokens)

decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)

# ---------------------------
# Use with your model
# ---------------------------

encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)
vocab_size = tokenizer.get_vocab_size()

print("Vocab size:", vocab_size)

# Encode dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:50])


# Encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])  # first few tokens

# split data into training and testing
n = int(0.1*len(data))
train_data = []
val_data = []
for i in range(10):
    start =n*i
    stop = int(0.9*n*(i+1))
    print(start, stop,n*(i+1))
    train_data += data[start:stop]
    val_data += data[stop:n*(i+1)]



def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # produce 4 random numbers that are between 0 and len(train_data)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def lr_lambda(step):
    warmup_steps = 1000
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))  # linear warmup
    # cosine decay after warmup
    progress = float(step - warmup_steps) / float(max(1, max_iters - warmup_steps))
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))  # cosine decay



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        # key is "what do I contain" and the query is "what do I need" ... whenever the dot-product of the query and
        # all the keys of the other tokens, when the key of token x and the query of token y "align", then they would have
        # high affinity, which will be represented as high value in the row x and column y ...
        # this affinity matrix will be the wei matrix
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        # note about masking here: the masking makes sure you have a triangular matrix, where the first row has one value
        # only, and the second 2 values, and so on. Given that, within each batch, columns represent tokens, this means that the
        # first token can only take information from itself, the second from itself and the first and so on ...
        # that is why we mask with zeros for the cells that are beyond the diagonal using .tril
        # the weights here represents affinities as a square matrix of T x T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        # v here is the value that will get aggregated for a particular token, given its key and query
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # create the affinity matrix multiple times
        self.ffwd = FeedFoward(n_embd) # think about the result
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # here you normalize x, then you apply the multihead attention, which is not how the original paper suggested,
        #   but it is ok and maybe better
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x





# embeddings are learnt vectors, whose values represent the word's (or letter's) relationship to other words (or letteres)
# it can be used  to find synonims, translation, classification ... etc
# when we talk about letters, it is more about the letter's context. in other words, it represents patterns of letter usage
# rather than semantic meaning. for example, t & h often appear together, so their embeddings may become related
# Embeddings are what gets 'learnt' in the context of LLM
# TOOD: check embedding of t and h .. and q and u ... and z

# the model starts by creating some sort of embedding of xb using self.token_embedding_table
# then, it matches the resulting numbers

# In the model, B is for batch, T for token, and C for channel (containing the embeddings, meaning that the channel
#   includes multiple embeddings per token, which might be understood as different dimensions of embeddings)

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) # this gives an indication about "what" the token is
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) # an indication about "where" the token is
        # position embeddings are important to the meaning of sentences, where
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

print(loss.item())


print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
