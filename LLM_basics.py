# Tutorial for building a chatGPT-like program
# start by showing that chatGPT is probabilistic not deterministic
# the paper is Attention is all you need
# the source of this tutorial is the following:
#   - https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing
#   - chatGPT
#   - https://jitter.video/ for gif creation
# Building a nano-GPT Model
''' This tutorial will show you how to build a small LLM, where the model creates a text based on the input text.'''

### Import libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests

torch.manual_seed(111)


# Load the required data. Here we use tinyShakespeare. You can use your own data if you want.
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url) # gt the contents of the url. This gives a response that includes response.text (among other things)
text = response.text

# check the downloaded text
print("length of dataset in characters: ", len(text),'\n ------------------------------------')
print(text[:500])


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


# We will tokenize the text: convert characters to integer values
stoi ={ch:i for i,ch in enumerate(chars)} # string to integer dictionary
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c]for c in s] # encoder function: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder function: take a list of integers, output a string
print('Encoding of "hello"           → ', encode("hole"))
print('Decoding of [46, 53, 50, 43]  ← ', decode([46, 53, 50, 43]))


# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100]) # the 1000 characters we looked at earier will to the GPT look like this

# split data into training and testing
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")



batch_size = 4 # how many independent sequences will we process in parallel?... will be referred to as B in the following code
block_size = 8 # what is the maximum context length for predictions? ... will be referred to as T in the following code

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # produce 4 random numbers that are between 0 and len(train_data)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y



xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")



# embeddings are learnt vectors, whose values represent the word's (or letter's) relationship to other words (or letteres)
# it can be used  to find synonims, translation, classification ... etc
# when we talk about letters, it is more about the letter's context. in other words, it represents patterns of letter usage 
# rather than semantic meaning. for example, t & h often appear together, so their embeddings may become related
# Embeddings are what gets 'learnt' in the context of LLM
# TOOD: check embedding of t and h .. and q and u ... and z


vocab_size
aa = nn.Embedding(65, 65)
aa(torch.tensor(8))

# the model starts by creating some sort of embedding of xb using self.token_embedding_table
# then, it matches the resulting numbers

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            # logits are the embiddings for a particular index (letter in this case).
            # to compute loss, softmax function is applied to give probabilities for each letter.
            # the embedding with highest probability is the predicted next letter.
            # if the probability of the target is 1, then the loss is 0. when probability is <1, loss increases and so on
            # here is what happens - simplified code:
            #    probs = softmax(logit)
            #    correct_letter_probs = probs[:,targets]
            #    loss = mean(-log(correct_letter_probs))
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # change the shape so that it coforms with what cross_entropy expects
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the index of the last letter only
            logits = logits[:, -1, :] # becomes (B, C), which is 
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample an index from the distribution: just returns an index from 0 to len(probs)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb) # this actually is similar to m.forward(xb, yb) 
print(logits.shape)
print(loss)

# idx is the starting point for the generation
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=15)[0].tolist()))


# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for steps in range(10000): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))
