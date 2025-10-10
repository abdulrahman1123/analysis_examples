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
import platform
if platform.system() == 'Darwin':
    import torch_directml
from tokenizers.processors import TemplateProcessing
import os
from nanogpt_utils import *
import wandb
wandb.login(key='a5bafd4256c35a962c34403b3a2ce9076f44d6c4')

torch.manual_seed(111)

config = wandb.config


# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel... will be referred to as B in the following code
block_size = 32 # what is the maximum context length for predictions ... will be referred to as T in the following code

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
if torch.cuda.is_available():
    device = 'cuda'
elif not platform.system()=='Darwin':
    device = 'cpu'
else:
    device = torch_directml.device()
eval_iters = 200
n_embd = 128
n_head = 3
n_layer = 3
n_embd = (n_embd//n_head)*n_head # inside the code, head_size is calculated as n_embd//n_head, which might give an error if the result are not a full integer
dropout = 0.1
clip_norm = 1.0
vocab_size = 12000
# ------------

base_dirs = [r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\documents\ML learning']
base_dir = [item for item in base_dirs if os.path.exists(item)][0]
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
#texts = [requests.get(url).text for url in [url1,url2,url3,url4,url5,url6,url7][0:1]] # gt the contents of the url. This gives a response that includes response.text (among other things)
#texts = [text[0:text.find('*** END OF THE PROJECT GUTENBERG')].replace('\r','').replace('\n\n','\r\r').replace('\n',' ').replace('\r\r','\n') for text in texts]
texts = [text[0:text.find('*** END OF THE PROJECT GUTENBERG')].replace('\r','') for text in texts]
text = '\n\n'.join(texts)

# check the downloaded text
print("length of dataset in characters: ", len(text),'\n ------------------------------------')
print(text[:500])


encode,decode, tokenizer, vocab_size = tokenize(text, vocab_size)

# Encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])  # first few tokens

train_data, val_data = train_test_split(data, 0.9)
train_data, val_data = train_data.to(device), val_data.to(device)


# -------------------------
# Initialize wandb
# -------------------------
# TOOD: maybe you will add lists of parameters
run = wandb.init(
    project="llm-hyperparameter-study",
    config={
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "block_size": block_size,
        "n_head": n_head,
        'eval_iters': eval_iters,
        "dropout": dropout,
        "n_layer": n_layer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "clip_norm": clip_norm
    }
)
config = run.config

model = BigramLanguageModel(config.vocab_size, config.n_embd, config.block_size,
                            config.n_head, config.dropout, config.n_layer, device)
m = model.to(device)
wandb.watch(m, log="all")  # tracks gradients, parameters, etc.
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(max_iters))

for iter in range(config.max_iters):

    # Evaluate periodically
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss(model, config.eval_iters, train_data, val_data, config.batch_size, config.block_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Log metrics
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "learning_rate": scheduler.get_last_lr()[0],
            "step": iter
        })

    # Training step
    xb, yb = get_batch(train_data, config.batch_size, config.block_size)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), config.clip_norm)
    optimizer.step()
    scheduler.step()


print(loss.item())


prompt = "To be, or not to be "
input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
out = m.generate(idx=input_ids, max_new_tokens=500)
print(tokenizer.decode(out[0].tolist()))

#idx = torch.zeros((1, 1), dtype=torch.long, device=device)
#out = m.generate(idx=idx, max_new_tokens=500)
#print(decode(out[0].tolist()))

