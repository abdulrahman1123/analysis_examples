'''
This code is modified from https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing,
which is explained in this video https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2409s
I modified it to take words as tokens
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import platform
from tokenizers.processors import TemplateProcessing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.processors import TemplateProcessing
import time
import matplotlib.patches as patches


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
        train_idx.append(idx[start:stop])
        val_idx.append(idx[stop:n * (i + 1)])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)

    train_data = data[train_idx]
    val_data = data[val_idx]

    return train_data, val_data

def train_model(model, max_iters, eval_iters, train_data, val_data, batch_size, eval_interval, block_size, device,optimizer,scheduler):
    for iter in range(max_iters):
        # Evaluate periodically
        if iter % 200 == 0:
            time.sleep(10)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, train_data, val_data,
                                   batch_size, block_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Training step
        xb, yb = get_batch(train_data, batch_size, block_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()


def tkn_pos_finder(text,tokenizer, spacing = 0.05):
    '''
    returns the positions of tokens in a sentence, as well as the tokens and token IDs
    '''
    text = ' '.join(text.split(' ')[0:10]) #make sure there is no more than 10 words
    encoded = tokenizer.encode(text)
    good_inds = [i for i in range(len(encoded.tokens)) if encoded.tokens[i] not in ['<EOS>','<BOS>']]
    token_ids = np.array(encoded.ids)[good_inds]
    tokens = [tokenizer.decode([item]) for item in token_ids]

    x_pos = [0]+np.cumsum([len(token) * 0.12+spacing for token in tokens]).tolist()
    x_widths = [x_pos[i+1] - x_pos[i]-spacing for i in range(len(x_pos)-1)]
    x_pos = x_pos[0:-1]
    return tokens, token_ids, x_pos, x_widths


def plot_line_tokens(text, ax, tokenizer, colors = ['#e6f3ff', '#fff0e6', '#e6ffe6', '#f0e6ff', '#fffae6'],
                     y_height=0.8, plot_ids = True, y_pos=0, spacing = 0):
    tokens, token_ids, x_pos, x_widths = tkn_pos_finder(text, tokenizer, spacing=spacing)

    for i, (token, token_id,x_p, x_w) in enumerate(zip(tokens, token_ids, x_pos,x_widths)):
        rect = patches.Rectangle((x_p, y_pos), x_w, y_height,linewidth=1.5, edgecolor='black',
                                 facecolor=colors[i % len(colors)], alpha=0.7)
        ax.add_patch(rect)

        # Add token text
        ax.text(x_p + x_w / 2, y_pos + y_height / 2,token, ha='center', va='center', fontsize=10,
                fontfamily='monospace', weight='bold')

        if plot_ids:
            ax.text(x_p + x_w / 2, y_pos - 0.2, f'{token_id}', ha='center', va='center', fontsize=8,
                    fontfamily='monospace', color='gray')
    max_x = x_p + x_w # to find the maximum x position for that line
    return max_x

def plot_sentence_tokens(text,tokenizer):
    fig, ax = plt.subplots( figsize=(7,1.5))
    max_x = plot_line_tokens(text, ax, tokenizer, spacing = 0.05)
    ax.text(-0.1, 0.4, 'Tokens: ', ha='right', va='center', fontsize=14, fontfamily='Calibri', weight='bold')
    ax.text(-0.1, -0.2 , 'IDs: ', ha='right', va='center', fontsize=14, color ='grey', fontfamily='Calibri', weight='bold')
    ax.set_xlim(-0.7, max_x + 0.1)
    ax.set_ylim(-0.1, 1.5)
    ax.set_aspect(0.6)
    ax.axis('off')  # Turn off axes
    plt.tight_layout()
