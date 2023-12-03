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


``` # Function to create mini-batches for training or validation.
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



## Step 3: Adding Positional Encodings
- **PosEncoding**: Adding positional information to the model with the `positional_encodings_table` in the `BigramLM` class.

## Step 4: Incorporating Adam Optimizer
- **AdamOptimization**: Initializing the Adam optimizer with `torch.optim.AdamW(model.parameters(), lr = lr_rate)`.

## Step 5: Introducing Self-Attention
- **OneHeadSelfAttention**: Incorporating a scaled dot product attention mechanism in the `SelfAttention` class.

## Step 6: Transitioning to Multi-Head Self-Attention
- **MultiHeadAttention**: Combining outputs from multiple `SelfAttention` heads in the `MultiHeadAttention` class.

## Step 7: Adding Feed-Forward Networks
- **FeedForward**: Implementing a feed-forward neural network with ReLU activation within the `FeedForward` class.

## Step 8: Formulating Blocks (Nx Layer)
- **TransformerBlocks**: Stacking transformer blocks using the `Block` class to create a deeper network architecture.

## Step 9: Adding Residual Connections
- **ResidualConnections**: Enhancing the `Block` class to include residual connections, improving learning efficiency.

## Step 10: Incorporating Layer Normalization
- **LayerNorm**: Normalizing layer outputs with `nn.LayerNorm(d_model)` in the `Block` class.

## Step 11: Implementing Dropout
- **Dropout**: To be added to the `SelfAttention` and `FeedForward` layers as a regularization method to prevent overfitting.

## Step 12: Scaling the Model
- **ScaleUp**: Increasing the complexity of the model by expanding `batch_size`, `block_size`, `d_model`, `d_k`, and `Nx`.

Throughout this project, we ensure that each component added aligns with the underlying principles of the Transformer model architecture. We aim for our Baby GPT to not only understand language patterns but also generate coherent text.

