import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1111)
B, T, C = 4, 8, 32  # batchsize, Time steps, Channels = dimension of mode, or teh number of features we are using to describe each token
print("Batch Size B, Time Steps T, C Number of Channels")

# create random integers
X = torch.randn(B,T,C)
print("Random (B, T, C) data proxy for embeddings", X)

# create lower triangular matrix T by T lower parts 1s and upper 0s
lower_tri = torch.tril(torch.ones(T,T))
print("initialization: lower triangle matrix: ", lower_tri)

# weight matrix T by T with 0s for illustration
att_scores = torch.zeros((T,T))
print("Initializing Attentian Matrix: ", lower_tri)

# filling zero weight matrix where 0 (upper part) with -inf and the rest with values in lower_tri
att_scores = att_scores.masked_fill(lower_tri == 0, float('-inf'))
print("Attention Matrix Masking Upper Triangle", att_scores)

# applying "SoftMax" transformation for normalization
att_scores = F.softmax(att_scores, dim=-1)
print("Attention Matrix after SoftMax")

#Multiplying embeddings and attention matrix "MatMul" 
output = att_scores @ X 
print("Dimension of Output", output.shape)
print("Output after MatMul",output)
