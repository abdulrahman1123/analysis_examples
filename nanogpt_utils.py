
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import platform
if platform.system() == 'Darwin':
    import torch_directml
from tokenizers.processors import TemplateProcessing
import os
from nanogpt_utils import *


def get_batch(data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,)) # produce 4 random numbers that are between 0 and len(train_data)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(model,val_iters, train_data, val_data, batch_size, block_size):
    out = {}
    model.eval()
    for data,data_n in zip([train_data, val_data],['train','val']):
        losses = torch.zeros(val_iters)
        for k in range(val_iters):
            X, Y = get_batch(data,batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[data_n] = losses.mean()
    model.train()
    return out


def get_lr_lambda(max_iters, warmup_steps=1000):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))  # linear warmup
        # cosine decay after warmup
        progress = float(step - warmup_steps) / float(max(1, max_iters - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))
    return lr_lambda



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size,n_embd,block_size,dropout):
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

    def __init__(self, num_heads, head_size,n_embd,block_size,dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_embd,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
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

    def __init__(self, n_embd, n_head,block_size,dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size,n_embd,block_size,dropout) # create the affinity matrix multiple times
        self.ffwd = FeedFoward(n_embd,dropout) # think about the result
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # here you normalize x, then you apply the multihead attention, which is not how the original paper suggested,
        #   but it is ok and maybe better
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size, n_embd, block_size, n_head, dropout, n_layer, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,block_size,dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) # this gives an indication about "what" the token is
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C) # an indication about "where" the token is
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
            idx_cond = idx[:, -self.block_size:]
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







def tokenize(text,vocab_size):
    # Initialize a ByteLevel BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )

    # ByteLevel pre-tokenizer (splits into byte-level pieces, can fully reconstruct text)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Train directly on your dataset string (in-memory, no saving to disk)
    tokenizer.train_from_iterator([text], trainer=trainer)

    # Define post-processing to add <BOS> and <EOS>
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> $B:1 <EOS>:1",
        special_tokens=[
            (tokenizer.token_to_id("<BOS>"), "<BOS>"),
            (tokenizer.token_to_id("<EOS>"), "<EOS>"),
        ],
    )

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
    return encode, decode, tokenizer, vocab_size


def train_test_split(data, train_ratio=0.9):
    # split data into training and testing
    n = int(0.1 * len(data))  # size of each chunk (10 parts)
    idx = torch.arange(len(data))

    train_idx = []
    val_idx = []

    for i in range(10):
        start = n * i
        stop = start + int(train_ratio * n)
        print(start, stop, n * (i + 1))
        train_idx.append(idx[start:stop])
        val_idx.append(idx[stop:n * (i + 1)])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)

    train_data = data[train_idx]
    val_data = data[val_idx]

    print('train data shape: ', train_data.shape, 'val data shape:', val_data.shape)
    return train_data, val_data









