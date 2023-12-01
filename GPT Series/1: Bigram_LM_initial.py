# check the input text, inspect the quality
with open('./GPT Series/input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# print first 100 characters
#print(text[:100])

# find the size of your data
#print("size of the dataset in #chars: ", len(text))

#------------------ Characterization and Vocabulary -------------------------#
## creating vocabulary from our data: list of characters sorted
# removing duplicate characters
unique_chars = set(text)
# pitting into list
list_unique_chars = list(unique_chars)
chars = sorted(list_unique_chars)
# size of vocabulary
vocab_size = len(chars)
# putting characters back into words
#print(''.join(chars))
#print(vocab_size)

#------------------------ Tokenization 1: Character tokenization -------------------------#
# transforming characters to integers: mapping each character to integer
chars_to_int = {c:i for i, c in enumerate(chars)}
int_to_chars = {i:c for i, c in enumerate(chars)} #creating dictionary to show for each char the integer order int:char
#print(int_to_chars)

# input: string
# output: list of integers
def encode(s):
    encoding = [chars_to_int[c] for c in s]
    return encoding

# input: list of integers
# output: list of integer
def decode(l):
    decoding = ''.join([int_to_chars[i] for i in l])
    return decoding

#print("--------- Char Tokenization ------------")
#print(encode("building simple LLM with GPT"))
#print(decode(encode("building simple LLM with GPT")))


# --------------------- Tokenization 2: Subword tokenization with tiktoken OPENAI ----------------------#
import tiktoken #used by openai for subword tokenization
enc = tiktoken.get_encoding("gpt2")
#print("number of tokens in OpenAI used tokenization tiktoken: ", enc.n_vocab)
#print("--------- Sub-word Tokenization with tiktoken ------------")
#print(enc.encode("building simple LLM with GPT"))
#print(enc.decode(enc.encode("building simple LLM with GPT")))


# -------------------- Tokenizing the entire data using PyTorch ------------#
import torch
# setting seed for reproducibility
torch.manual_seed(1111)
# encoding the text using tiktoken and transform it to tensors 
# to go from high dimensional space to single vector (1 very large vector)
#data = torch.tensor(enc.encode(text), dtype= torch.long)
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)


#---------------------------- Data Preparation -----------------------------#

# splitting the data into train and test
split_90perc = int(0.9*len(data))
#print("Size of the data: ", len(data))
#print("Training size: ", split_90perc)
#print("Validation or Test size: ", len(data) - split_90perc)
train_data = data[:split_90perc]
valid_data = data[split_90perc:]


# block size or the model size: maximum context length the model sees each time for predictions
block_size = 8
#print(train_data[:block_size+1])
#print("Block Size is: ", block_size)


# creating mini-batches for efficiency
# batch_size: how many independent sequences we will process in parallel?
batch_size = 4
# function for creating mini batches
def get_batch(split):
    # if training split 
    if split == "train":
        data = train_data
    # or validation split
    else:
        data = valid_data
    # Generate 'batch_size' random starting indices for data blocks. 
    # Each index is within the range [0, len(data) - block_size) to ensure 
    # each block has 'block_size' elements without exceeding data bounds.
    ix = torch.randint(len(data)-block_size, (batch_size,))
    # we are going to use the sequences with block_size (X) to predict the next token(Y)
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack of sequences (batch size rows, block size columns) to use to predict Y
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # stack of sequences from next token onwards in compared to x , what is to be predcited (real y)
    return x,y

# batch x_s, batch y_s
xb, yb = get_batch("train")
#print('input sequences')
#print("shape: ", xb.shape)
#print(xb)

#print('target sequences to predict')
#print("shape: ",yb.shape)
#print(yb)


# visualizing stacked input sequences and their corresponding target output
# per batch
# per time step (each in block_size) as we grow the context
for batch in range(batch_size): # number of batches
    #print("Batch number: ", batch)
    for t in range(block_size):
        #print("Time Step: ", t)
        # taking the batch batch_num and the time_step t
        context = xb[batch, :t+1]
        target = yb[batch, t]
        #print(f"When input is {context.tolist()} the next word to be predicted is: {target} \n")


# --------------------------- Bigram Language Model ------------------------#
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1111)



# creating class of BigramML for Biagram model: type of N-Gream models
class BigramLM(nn.Module):
    
    #initializing the class, will be called every time when we create an instance of this class
    def __init__(self, vocab_size):
        # super class initialization
        # In OOP, it's common practice to initialize the base class when subclassing. 
        # This ensures that the base class is properly set up before you add your custom behavior.
        super().__init__()
        # Layer 1: Embedding Layer (embedding is of size vocab by vocab)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    # Function to generate predictions (logits) for the next token in a sequence and, if targets are provided, to calculate the loss based on these predictions.
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # log conditional probabilities        
        logits = self.token_embedding_table(idx) # (B,T,C)

        # no target no loss
        # to evaluate how well the model is predicting the next token
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # transforming our array from B,T,C to B*T,C done for Pytorch functionality
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # function that generates updated tokens as the new tokens are added per time step, per batch
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context 
        for _ in range(max_new_tokens):
            # get the predictions corresponding to this input, log probabilitis, loss just for placeholder not used
            logits, loss = self(idx)
            # keep only the last time step, we need the logits of the last time step as this is bigram
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities from the logits (from log probabilities to value between 0,1)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from multinomial distribution to randomly pick a token based on the calculated probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) only single next value selected
            # append sampled token to the current sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# an instance of the BigramLM class
model = BigramLM(vocab_size)
# calls for forward method to do forwards pass in Bigram Model
# returns 
# 1: logits, the raw, unnormalized predictions for the next tokens
# 2: loss, which is a measure of how well the model's predictions match the actual next tokens (yb)
logits, loss = model(xb, yb)
print(logits.shape)
print("negative loss",loss)
# calls for generate method
# creates a starting point for the generation. This is typically a tensor of zeros, which might represent a start-of-sequence token or simply a placeholder to start generation.
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=512)[0].tolist()))



#----------------------------- Building PyTorch optimizer using Adam -------------------------------#
lr_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr = lr_rate)

batch_size = 32
num_train_iter = 10000
# for every step in Adam optimizer (100 iterations)
for step in range(num_train_iter):
    # sampling a mini batch of data
    xb, yb = get_batch("train")

    # Forward Pass: obtaining raw predictions (logits) and calculate loss(cross-entropy) using our Bi-gram based model
    logits, loss = model(xb, yb)
    # Zeroing Gradients: Before computing the gradients, existing gradients are reset to zero. This is necessary because gradients accumulate by default in PyTorch.
    optimizer.zero_grad(set_to_none=True)
    # Backward Pass or Backpropogation: Computing Gradients
    loss.backward()
    # Updating the Model Parameters
    optimizer.step()
    #printing the Loss

print(loss.item())

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=512)[0].tolist()))
