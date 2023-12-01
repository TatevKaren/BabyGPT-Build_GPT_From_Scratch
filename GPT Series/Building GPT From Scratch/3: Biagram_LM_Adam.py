import tiktoken #used by openai for subword tokenization
import torch
print(torch.cuda.is_available())
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1111)

#------------------------------- HyperParameters ----------------------------#
batch_size = 32
block_size = 8
num_iter = 10000
eval_interval = 500 # how often to compute the loss
lr_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#----------------------------- Loading Data ---------------------------------#
# check the input text, inspect the quality
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

#------------------ Characterization and Vocabulary -------------------------#
unique_chars = set(text)
list_unique_chars = list(unique_chars)
chars = sorted(list_unique_chars)
vocab_size = len(chars)


#------------------------ Tokenization 1: Character tokenization -------------------------#
# transforming characters to integers: mapping each character to integer
chars_to_int = {c:i for i, c in enumerate(chars)}
int_to_chars = {i:c for i, c in enumerate(chars)} #creating dictionary to show for each char the integer order int:char

# strings to integer
def encode(s):
    encoding = [chars_to_int[c] for c in s]
    return encoding

# list to strings
def decode(l):
    decoding = ''.join([int_to_chars[i] for i in l])
    return decoding



# -------------------- Tokenizing the entire data using PyTorch ------------#
data = torch.tensor(encode(text), dtype=torch.long)




#---------------------------- Data Preparation -----------------------------#

# splitting the data into train and test
split_90perc = int(0.9*len(data))
#print("Size of the data: ", len(data))
#print("Training size: ", split_90perc)
#print("Validation or Test size: ", len(data) - split_90perc)
train_data = data[:split_90perc]
valid_data = data[split_90perc:]


torch.manual_seed(1111)
# function for creating mini batches
def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = valid_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack of sequences (batch size rows, block size columns) to use to predict Y
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stack of sequences from next token onwards in compared to x , what is to be predcited (real y)
    # adding this line to give the data to device in case cude:GPU
    x, y = x.to(device), y.to(device)
    return x,y


# --------------------------- Bigram Language Model ------------------------#
class BigramLM(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Layer 1: Embedding Layer (embedding is of size vocab by vocab)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):     
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # function that generates updated tokens as the new tokens are added per time step, per batch
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context 
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) only single next value selected
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLM()
# moving model paramters to device
model_device = model.to(device)


#-------------------------------------- Estimating Loss ---------------------------------------#
# less noisy loss: average loss across every X batches
# making sure PyTorch doesn't store the parameters backwards
@torch.no_grad()
def estimate_loss():
    result = {}
    # setting model in evaluation state
    model.eval()
    for split in ['train', 'valid_date']:
        losses = torch.zeros(eval_iters)
        for e in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            # storing each iterations loss
            losses[e] = loss.item()
        result[split] = losses.mean()
    # setting back to training state
    model.train()
    return result




#----------------------------- Building PyTorch optimizer using Adam -------------------------------#
optimizer = torch.optim.AdamW(model.parameters(), lr = lr_rate)

for iter in range(num_iter):
    # estimating the loss for per X interval
    if iter % eval_interval == 0:
       losses = estimate_loss()
       print(f"step {iter}: train loss is {losses['train']:.5f} and validation loss is {losses['valid_date']:.5f}")
    # sampling a mini batch of data
    xb, yb = get_batch("train")

    # Forward Pass 
    logits, loss = model(xb, yb)
    # Zeroing Gradients: Before computing the gradients, existing gradients are reset to zero. This is necessary because gradients accumulate by default in PyTorch.
    optimizer.zero_grad(set_to_none=True)
    # Backward Pass or Backpropogation: Computing Gradients
    loss.backward()
    # Updating the Model Parameters
    optimizer.step()
    #printing the Loss

context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


