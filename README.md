# Baby GPT - Simple Language Modeling Project

## Detailed Steps in Model Architecture
Baby GPT is an exploratory project designed to incrementally build a GPT-like language model. The project begins with a simple Bigram Model and gradually incorporates advanced concepts from the Transformer model architecture.

![Transformer Model Architecture](./GPT%20Series/Images/AttentionIsAllYouNeed.png)


## Hyperparameters


The model's performance is tuned using the following hyperparameters:

- `batch_size`: The number of sequences processed in parallel during training
- `block_size`: The length of the sequences being processed by the model
- `d_model`: The number of features in the model (the size of the embeddings)
- `d_k`: The number of features per attention head. 
- `num_iter`: The total number of training iterations the model will run
- `Nx`: The number of transformer blocks, or layers, in the model. 
- `eval_interval`: The interval at which the model's loss is computed and evaluated
- `lr_rate`: The learning rate for the Adam optimizer
- `device`: Automatically set to `'cuda'` if a compatible GPU is available, otherwise defaults to `'cpu'`.
- `eval_iters`: The number of iterations over which to average the evaluation loss
- `h`: The number of attention heads in the multi-head attention mechanism
- `dropout_rate`: The dropout rate used during training to prevent overfitting


These hyperparameters were carefully chosen to balance the model's ability to learn from the data without overfitting and to manage computational resources effectively.

<br>


## Decoder Only for GPT


## CPU vs GPU Model Example



| Hyperparameter   | CPU Model           | GPU Model           |
|------------------|---------------------|---------------------|
| `device`         | 'cpu'               | 'cuda' if available, else 'cpu' |
| `batch_size`     | 16                  | 64                  |
| `block_size`     | 8                   | 256                 |
| `num_iter`       | 10000               | 10000               |
| `eval_interval`  | 500                 | 500                 |
| `eval_iters`     | 100                 | 200                 |
| `d_model`        | 16                  | 512                 |
| `d_k`            | 4                   | 16                  |
| `Nx`             | 2                   | 6                   |
| `dropout_rate`   | 0.2                 | 0.2                 |
| `lr_rate`        | 0.005 (5e-3)        | 0.001 (1e-3)        |
| `h`              | 2                   | 6                   |



<br>



# Step 1: Data Preparation Including Tokenization
- **LoadData**: `open('./GPT Series/input.txt', 'r', encoding = 'utf-8')`
- **BuildVocab**: Creating a vocabulary dictionary with `chars_to_int` and `int_to_chars`.
- **CharacterTokenization**: Converting strings to integers with the `encode` function and back with the `decode` function.
- **DataSplit**: Dividing the dataset into training (`train_data`) and validation (`valid_data`) sets.



# Step 2: Building a Simple Bigram Language Model (Initial Model)
- **Mini Batch**: The `get_batch` function prepares data in mini-batches for training.
- **BigramModel**: Defines the model architecture in the `BigramLM` class.
- **TrainModel**: Outlines the training procedure using the Adam optimizer and loss estimation.

## Mini Batch Technique
Mini-batching is a technique in machine learning where the training data is divided into small batches. Each mini-batch is processed separately during model training. This approach helps in:

- Efficient Use of Memory: By not loading the entire dataset into memory at once, it reduces computational overhead.
- Faster Convergence: Processing data in batches can lead to faster convergence compared to processing each data point individually.
- Improved Generalization: Mini-batches introduce noise into the training process, which can help the model generalize better to unseen data.


```python
# Function to create mini-batches for training or validation.
def get_batch(split):
    # Select data based on training or validation split.
    data = train_data if split == "train" else valid_data

    # Generate random start indices for data blocks, ensuring space for 'block_size' elements.
    ix = torch.randint(len(data)-block_size, (batch_size,))

    # Create input (x) and target (y) sequences from data blocks.
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # Move data to GPU if available for faster processing.
    x, y = x.to(device), y.to(device)

    return x, y
```

## How to choose your Batch Size?


| Factor           | Small Batch Size                                       | Large Batch Size                                   |
|------------------|--------------------------------------------------------|----------------------------------------------------|
| **Gradient Noise**       | Higher (more variance in updates)                       | Lower (more consistent updates)                    |
| **Convergence**          | Tends to explore more solutions, including flatter minima | Often converges to sharper minima                   |
| **Generalization**       | Potentially better (due to flatter minima)               | Potentially worse (due to sharper minima)           |
| **Bias**                 | Lower (less likely to overfit to training data patterns) | Higher (may overfit to training data patterns)      |
| **Variance**             | Higher (due to more exploration in solution space)       | Lower (due to less exploration in solution space)   |
| **Computational Cost**   | Higher per epoch (more updates)                           | Lower per epoch (fewer updates)                     |
| **Memory Usage**         | Lower                                                    | Higher                                              |




## Estimating Loss (Negative Loss Likelihood or Cross Entropy)

The ```estimate_loss``` function calculates the average loss for the model over a specified number of iterations (eval_iters). It's used to assess the model's performance without affecting its parameters. The model is set to evaluation mode to disable certain layers like dropout for a consistent loss calculation. After computing the average loss for both training and validation data, the model is reverted to training mode. This function is essential for monitoring the training process and making adjustments if necessary.


```python
@torch.no_grad()  # Disables gradient calculation to save memory and computations
def estimate_loss():
    result = {}  # Dictionary to store the results
    model.eval()  # Puts the model in evaluation mode

    # Iterates over the data splits (training and validation)
    for split in ['train', 'valid_date']:
        # Initializes a tensor to store the losses for each iteration
        losses = torch.zeros(eval_iters)

        # Loops over the number of iterations to calculate the average loss
        for e in range(eval_iters):
            X, Y = get_batch(split)  # Fetches a batch of data
            logits, loss = model(X, Y)  # Gets the model outputs and computes the loss
            losses[e] = loss.item()  # Records the loss for this iteration

        # Stores the mean loss for the current split in the result dictionary
        result[split] = losses.mean()

    model.train()  # Sets the model back to training mode
    return result  # Returns the dictionary with the computed losses
```

<br>


# Step 3: Adding Positional Encodings
**Positional Encoding**: Adding positional information to the model with the `positional_encodings_table` in the `BigramLM` class.
We add Positional Encodings to the embeddings of our characters as in the transformer architecture.

<br>


# Step 4: Incorporating Adam W Optimizer
Here we set up and use the AdamW optimizer for training a neural network model in PyTorch. The Adam optimizer is favored in many deep learning scenarios because it combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. Adam computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients like RMSProp, Adam also keeps an exponentially decaying average of past gradients, similar to momentum. This enables the optimizer to adjust the learning rate for each weight of the neural network, which can lead to more effective training on complex datasets and architectures.


```AdamW``` modifies the way weight decay is incorporated into the optimization process, addressing an issue with the original Adam optimizer where the weight decay is not well separated from the gradient updates, leading to suboptimal application of regularization. Using AdamW can sometimes result in better training performance and generalization to unseen data. We chose AdamW for its ability to handle weight decay more effectively than the standard Adam optimizer, potentially leading to improved model training and generalization.


```python

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
```

<br>


# Step 5: Introducing Self-Attention

Self-Attention is a mechanism that allows the model to weigh the importance of different parts of the input data differently. It is a key component of the Transformer architecture, enabling the model to focus on relevant parts of the input sequence for making predictions.

- **Dot-Product Attention**: A simple attention mechanism that computes a weighted sum of values based on the dot product between queries and keys.
  
- **Scaled Dot-Product Attention**: An improvement over the dot-product attention that scales down the dot products by the dimensionality of the keys, preventing gradients from becoming too small during training.

- **OneHeadSelfAttention**: Implementation of a single-headed self-attention mechanism that allows the model to attend to different positions of the input sequence. The `SelfAttention` class showcases the intuition behind the attention mechanism and its scaled version.

![Transformer Model Architecture](./GPT%20Series/Images/AttentionMechanism.png)


Each corresponding model in the Baby GPT project incrementally builds upon the previous one, starting with the intuition behind the Self-Attention mechanism, followed by practical implementations of dot-product and scaled dot-product attentions, and culminating in the integration of a one-head self-attention module.


```python
class SelfAttention(nn.Module):
    """Self Attention (One Head)"""
    """ d_k = C """
    def __init__(self, d_k):
        super().__init__() #superclass initialization for proper torch functionality
        # keys 
        self.keys = nn.Linear(d_model, d_k, bias = False)
        # queries
        self.queries = nn.Linear(d_model, d_k, bias = False)
        # values
        self.values = nn.Linear(d_model, d_k, bias = False)
        # buffer for the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, X):
        """Computing Attention Matrix"""
        B, T, C = X.shape
        # Keys matrix K
        K = self.keys(X) # (B, T, C)
        # Query matrix Q
        Q = self.queries(X) # (B, T, C)
        # Scaled Dot Product
        scaled_dot_product = Q @ K.transpose(-2,-1) * 1/math.sqrt(C) # (B, T, T)
        # Masking upper triangle
        scaled_dot_product_masked = scaled_dot_product.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        # SoftMax transformation
        attention_matrix = F.softmax(scaled_dot_product_masked, dim=-1) # (B, T, T)
        # Weighted Aggregation
        V = self.values(X) # (B, T, C)
        output =  attention_matrix @ V # (B, T, C)
        retur
```


The `SelfAttention` class represents a fundamental building block of the Transformer model, encapsulating the self-attention mechanism with a single head. Here's an insight into its components and processes:

- **Initialization**: The constructor `__init__(self, d_k)` initializes the linear layers for keys, queries, and values, all with the dimensionality `d_k`. These linear transformations project the input into different subspaces for subsequent attention calculations.

- **Buffers**: `self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))` registers a lower triangular matrix as a persistent buffer that is not considered a model parameter. This matrix is used for masking in the attention mechanism to prevent future positions from being considered in each calculation step (useful in decoder self-attention).

- **Forward Pass**: The `forward(self, X)` method defines the computation performed at every call of the self-attention module


<br>

# Step 6: Transitioning to Multi-Head Self-Attention
**MultiHeadAttention**:Combining outputs from multiple `SelfAttention` heads in the `MultiHeadAttention` class. The MultiHeadAttention class is an extended implementation of the self-attention mechanism with one head from previous step, but now multiple attention heads operate in parallel, each focusing on different parts of the input. 

```python

class MultiHeadAttention(nn.Module):
    """Multi Head Self Attention"""
    """h: #heads"""
    def __init__(self, h, d_k):
        super().__init__()
        # initializing the heads, we want h times attention heads wit size d_k
        self.heads = nn.ModuleList([SelfAttention(d_k) for _ in range(h)])
        # adding linear layer to project the concatenated heads to the original dimension
        self.projections = nn.Linear(h*d_k, d_model)
        # adding dropout layer
        self.droupout = nn.Dropout(dropout_rate)
    
    def forward(self, X):
        # running multiple self attention heads in parallel and concatinate them at channel dimension
        combined_attentions = torch.cat([h(X) for h in self.heads], dim = -1)
        # projecting the concatenated heads to the original dimension
        combined_attentions = self.projections(combined_attentions)
        # applying dropout
        combined_attentions = self.droupout(combined_attentions)
        return combined_attentions
    
    
```

# Step 7: Adding Feed-Forward Networks
**FeedForward**: Implementing feed-forward neural network with ReLU activation within the `FeedForward` class. To add this fully connected feed-forward to our model as in original Transformer Model.
  
```python
class FeedForward(nn.Module):
    """FeedForward Layer with ReLU activation function"""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            # 2 linear layers with ReLU activation function
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_rate)
        )
    def forward(self, X):
        # applying the feedforward layer
        
        return self.net(X)
```
  

# Step 8: Formulating Blocks (Nx in Model)
**TransformerBlocks**: Stacking transformer blocks using the `Block` class to create a deeper network architecture.Depth and Complexity: In neural networks, depth refers to the number of layers through which data is processed. Each additional layer (or block, in the case of Transformers) allows the network to capture more complex and abstract features of the input data.

Sequential Processing: Each Transformer block processes the output of its preceding block, gradually building a more sophisticated understanding of the input. This sequential processing allows the network to develop a deep, layered representation of the data.
Components of a Transformer Block

- Multi-Head Attention: Central to each block, it processes the input by focusing on different parts of the sequence simultaneously. This parallel processing is key to the model's ability to understand complex data relationships.
- Feed-Forward Network: It further processes the data after attention, adding another layer of complexity.
- Residual Connections: They help maintain the flow of information across the network, preventing the loss of input data through layers and aiding in combating the vanishing gradient problem.
- Layer Normalization: Applied before each major component, it stabilizes the learning process, ensuring smooth training across deep layers.


```python
# ---------------------------------- Blocks ----------------------------------#
class Block(nn.Module):
    """Multiple Blocks of Transformer"""
    def __init__(self, d_model, h):
        super().__init__()
        d_k = d_model // h
        # Layer 4: Adding Attention layer
        self.attention_head = MultiHeadAttention(h, d_k) # h heads of d_k dimensional self-attention
        # Layer 5: Feed Forward layer
        self.feedforward = FeedForward(d_model)
        # Layer Normalization 1
        self.ln1 = nn.LayerNorm(d_model)
        # Layer Normalization 2
        self.ln2 = nn.LayerNorm(d_model)
    
    # Adding additional X for Residual Connections
    def forward(self,X):
        X = X + self.attention_head(self.ln1(X))
        X = X + self.feedforward(self.ln2(X))
        return X
  ```
<br>


# Step 9: Adding Residual Connections
**ResidualConnections**: Enhancing the `Block` class to include residual connections, improving learning efficiency.
Residual Connections, also known as skip connections, are a critical innovation in the design of deep neural networks, particularly in Transformer models. They address one of the primary challenges in training deep networks: the vanishing gradient problem. 

```python
    # Adding additional X for Residual Connections
    def forward(self,X):
        X = X + self.attention_head(self.ln1(X))
        X = X + self.feedforward(self.ln2(X))
        return X
```


# Step 10: Incorporating Layer Normalization
**LayerNorm**: Adding Layer Normalization to our Transformer.Normalizing layer outputs with `nn.LayerNorm(d_model)` in the `Block` class.

![Transformer Model Architecture](./GPT%20Series/Images/LayerNormalization.png)


```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # orward pass calculaton
        xmean = x.mean(1, keepdim=True)  # layer mean
        xvar = x.var(1, keepdim=True)  # layer variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta      
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

```



# Step 11: Implementing Dropout
**Dropout**: To be added to the `SelfAttention` and `FeedForward` layers as a regularization method to prevent overfitting. We add drop-out to:
- Self Attention Class
- Multi-Head Self Attention
- FeedForward 
  

# Step 12: Scaling the Model: NVIDIA CUDA for Using GPU
**ScaleUp**: Increasing the complexity of the model by expanding `batch_size`, `block_size`, `d_model`, `d_k`, and `Nx`. You will need CUDA toolkit as well as machine with NVIDIA GPU to train and test this bigger model. 

If you want to try out CUDA for GPU acceleration, ensure that you have the appropriate version of PyTorch installed that supports CUDA. 

```python
import torch
torch.cuda.is_available()
```

You can do this by specifying the CUDA version in your PyTorch installation command, like in command line:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

