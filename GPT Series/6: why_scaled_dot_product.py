import torch
import math
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(2222)
B, T, C = 4, 8, 32  # batchsize, Time steps (or number of tokens), Channels = dimension of mode, or teh number of features we are using to describe each token
print("Batch Size B, Time Steps T, C Number of Channels")

# create random integers
X = torch.randn(B,T,C)

# the head size, the number of features to include in each head when finding attentions
d_k = 16 

K = torch.randn(B, T, d_k)
Q = torch.randn(B, T, d_k)

print(K.var())
print(Q.var())

dot_product = Q @ K.transpose(-2,-1)
print("Dot Product: ", dot_product.var())

scaled_dot_product = dot_product * 1/math.sqrt(d_k)
print("Scaled Dot Product", scaled_dot_product.var(), "\n")


print("Soft Max on Dot Product", F.softmax(dot_product[0][0]), "\n")

print("Soft Max on Scaled Dot Product", F.softmax(scaled_dot_product[0][0]), "\n")



