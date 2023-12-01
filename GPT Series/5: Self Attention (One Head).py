import torch
import math
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(2222)
B, T, C = 4, 8, 32  # batchsize, Time steps (or number of tokens), Channels = dimension of mode, or teh number of features we are using to describe each token
print("Batch Size B, Time Steps T, C Number of Channels")

# create random integers
X = torch.randn(B,T,C)
print("Random (B, T, C) data proxy for embeddings", X)

# the head size, the number of features to include in each head when finding attentions
d_k = 16 

# ---------------------------- Keys K and Queries Q ------------------------------#
# Keys using C or model size (d_model) number of features for each token
keys = nn.Linear(C, d_k, bias = False)
# Queries with same size as Keys
queries = nn.Linear(C, d_k, bias = False)
# Values
values = nn.Linear(C, d_k, bias = False)


# Keys Matrix K (K' in official Transformer paper)
K = keys(X) # (B, T, head_size)
# Query Matrix Q (K' in official Transformer paper)
Q = queries(X) # (B, T, head_size)
# Value Matrix V (V' in official Transformer paper)
V = values(X) # (B, T, head_size)



# -------------------------------- Dot Product, Masking, SoftMax ----------------------------------#
# computing the "Dot Product"
dot_product = Q @ K.transpose(-2,-1) # (B, T, T)
scaled_dot_product = dot_product * 1/math.sqrt(d_k) # (B, T, T)
# every Batch B we get (T, T) attention matrix with dependencies between each pair of tokens
# create lower triangular matrix T by T lower parts 1s and upper 0s
lower_tri = torch.tril(torch.ones(T,T))
# "Masking": filling zero weight matrix where 0 (upper part) with -inf and the rest with values in lower_tri
masked_dot_product = scaled_dot_product.masked_fill(lower_tri == 0, float('-inf'))
# "SoftMax" transformation for normalization
att_scores = F.softmax(masked_dot_product, dim=-1)
# Multiplying embeddings and attention matrix "MatMul" 
#output = att_scores @ X
output  = att_scores @ V

#print("Dimension of Output", output.shape)
#print("Output after MatMul",output)

#print(att_scores[0])
#print(output[0])

