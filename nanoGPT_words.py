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
from tokenizers.processors import TemplateProcessing
import os
from nanogpt_utils import *
#import wandb
#wandb.login(key='a5bafd4256c35a962c34403b3a2ce9076f44d6c4')
#config = wandb.config

torch.manual_seed(111)


base_dirs = ['/Users/abdelrahmansawalma/Downloads/LLMs',
             r"\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\downloads\LLMs"]
base_dir = [item for item in base_dirs if os.path.exists(item)][0]

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel... will be referred to as B in the following code
block_size = 32 # what is the maximum context length for predictions ... will be referred to as T in the following code

max_iters = 10000
eval_interval = 500
learning_rate = 2e-4
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except:
        device = torch.device("cpu")
eval_iters = 200
n_embd = 128
n_head = 3
n_layer = 3
n_embd = (n_embd//n_head)*n_head # inside the code, head_size is calculated as n_embd//n_head, which might give an error if the result are not a full integer
dropout = 0.1
clip_norm = 1.0
vocab_size = 20000
# ------------

#base_dirs = [r'\\klinik.uni-wuerzburg.de\homedir\userdata11\Sawalma_A\data\documents\ML learning']
#base_dir = [item for item in base_dirs if os.path.exists(item)][0]
# Load the required data. Here we use tinyShakespeare. You can use your own data if you want.
url1 = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
url1='https://www.gutenberg.org/cache/epub/100/pg100.txt' # The Complete Works of William Shakespeare
url2 = 'https://www.gutenberg.org/cache/epub/11/pg11.txt' # Alice in wonderland
url3 = 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt' # pride and prejudice
url4 = 'https://www.gutenberg.org/cache/epub/2701/pg2701.txt' # moby dick
url5 = 'https://www.gutenberg.org/cache/epub/84/pg84.txt' # Frankenstein; Or, The Modern Prometheus
url6 = 'https://www.gutenberg.org/cache/epub/145/pg145.txt' # Middlemarch
url7 = 'https://www.gutenberg.org/cache/epub/67979/pg67979.txt' # The Blue Castle
texts = [requests.get(url).text for url in [url1,url2,url3,url4,url5,url6,url7]] # gt the contents of the url. This gives a response that includes response.text (among other things)
texts = [texts[0]]
#texts = [requests.get(url).text for url in [url1,url2,url3,url4,url5,url6,url7][0:1]] # gt the contents of the url. This gives a response that includes response.text (among other things)
#texts = [text[0:text.find('*** END OF THE PROJECT GUTENBERG')].replace('\r','').replace('\n\n','\r\r').replace('\n',' ').replace('\r\r','\n') for text in texts]
texts = [text[0:text.find('*** END OF THE PROJECT GUTENBERG')].replace('\r','') for text in texts]
text = '\n\n'.join(texts)
text = text[text.find('THE SONNETS\n\n')::]
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
if False:
    # TOOD: maybe you will add lists of parameters
    sweep_config = {
        'method': 'grid',
        'parameters':{
            "vocab_size": {"values":[100,500,1000, 2000]},
            "n_embd": {"values":[n_embd]},
            "block_size": {"values":[block_size]},
            "n_head": {"values":[n_head]},
            'eval_iters': {"values":[eval_iters]},
            "dropout": {"values":[dropout]},
            "n_layer": {"values":[n_layer]},
            "learning_rate": {"values":[learning_rate]},
            "batch_size": {"values":[batch_size]},
            "max_iters": {"values":[max_iters]},
            "eval_interval": {"values":[eval_interval]},
            "clip_norm": {"values":[clip_norm]}
        }
    }



    # 2. Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="llm-HS")


    # 3. Define your training function
    def train():
        run = wandb.init()
        config = run.config

        # Create model
        model = BigramLanguageModel(
            config.vocab_size, config.n_embd, config.block_size,
            config.n_head, config.dropout, config.n_layer, device
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(max_iters=config.max_iters))

        for iter in range(config.max_iters):
            # Evaluate periodically
            if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
                losses = estimate_loss(model, config.eval_iters, train_data, val_data,
                                       config.batch_size, config.block_size)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                wandb.log({
                    "train_loss": losses['train'],
                    "val_loss": losses['val'],
                    "step": iter
                })

            # Training step
            xb, yb = get_batch(train_data, config.batch_size, config.block_size)
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()
            scheduler.step()
        run.finish()

    wandb.agent(sweep_id, train)


    api = wandb.Api()
    sweep = api.sweep("abdulrahman-sawalma/llm-HS/tyvvin2p")
    runs = sorted(sweep.runs, key=lambda run: run.summary.get('val_loss', float('inf')))

    # Get the best run (lowest validation loss)
    best_run = runs[0]
    print(f"Best run: {best_run.name} with val_loss: {best_run.summary['val_loss']}")




batch_size = 64
block_size = 256
learning_rate = 2e-4
n_embd = 256
n_head = 4
n_layer = 4
n_embd = (n_embd//n_head)*n_head # inside the code, head_size is calculated as n_embd//n_head, which might give an error if the result are not a full integer


model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, dropout, n_layer, device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(max_iters=max_iters, warmup_steps=200))


for iter in range(max_iters):
    # Evaluate periodically
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    scheduler.step()




prompt = "To be, or not to be "
input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
out = model.generate(idx=input_ids, max_new_tokens=100)
print(tokenizer.decode(out[0].tolist()))

model_name = f'model_batch{batch_size}_vocab{vocab_size}_nembed{n_embd}_block{block_size}_nhead{n_head}_n_layer{n_layer}'
model_path = os.path.join(base_dir,f'{model_name}.pt')

#save the model
torch.save(model.state_dict(), model_path)
print(f"Model saved as: {model_name}.pt")


#load the model
model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, dropout, n_layer, device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()