# Baby GPT - Simple Language Modeling Project

## Detailed Steps in Model Architecture
Baby GPT is an exploratory project designed to incrementally build a GPT-like language model. The project begins with a simple Bigram Model and gradually incorporates advanced concepts from the Transformer model architecture.

## Step 1: Data Preparation
- **LoadData**: `open('./GPT Series/input.txt', 'r', encoding = 'utf-8')`
- **BuildVocab**: Creating a vocabulary dictionary with `chars_to_int` and `int_to_chars`.
- **CharacterTokenization**: Converting strings to integers with the `encode` function and back with the `decode` function.
- **DataSplit**: Dividing the dataset into training (`train_data`) and validation (`valid_data`) sets.

## Step 2: Building a Simple Bigram Language Model (Initial Model)
- **Batching**: The `get_batch` function prepares data in mini-batches for training.
- **BigramModel**: Defines the model architecture in the `BigramLM` class.
- **TrainModel**: Outlines the training procedure using the Adam optimizer and loss estimation.

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


