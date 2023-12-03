# Baby GPT - Simple Language Modeling Project

## Detailed Steps in Model Architecture
Baby GPT is an exploratory project designed to incrementally build a GPT-like language model. The project begins with a simple Bigram Model and gradually incorporates advanced concepts from the Transformer model architecture.

## Step 1: Data Preparation
1. **Loading and Inspecting Data**: Examination of the dataset for model training.
2. **Characterization and Vocabulary Building**: Generating a dictionary to map characters to integers.
3. **Tokenization**: 
   - Character Tokenization: Converting characters to numerical representations.
   - Subword Tokenization (Optional): Implementing a more advanced tokenization approach similar to GPT-2.
4. **Data Splitting**: 
   - Train/Test Split: Dividing the dataset into training and validation subsets.

## Step 2: Building a Simple Bigram Language Model (Initial Model)
1. **Creating Mini-Batches**: Implementation of a `getBatch` function for processing data in mini-batches.
2. **Model Definition**: Development of a class for the Bigram Language Model (`Bigram_lm_initial.py`) that calculates conditional probabilities as logits, and if targets are present, computes the loss as well.
3. **Model Training**: Procedures for training the initial Bigram model to optimize its parameters.

## Step 3: Adding Positional Encodings
1. **Understanding Positional Encodings**: Explanation of how positional encodings provide sequence context in transformer models.
2. **Implementation**: Integration of positional encodings as the second layer in the transformer model architecture.

## Step 4: Incorporating Adam Optimizer
1. **Optimizer Integration**: Adding the Adam optimizer to the model (`Bigram_lm_adam.py`) to improve training efficiency and prevent overfitting.

## Step 5: Introducing Self-Attention
1. **Single Head Self-Attention**: Establishing the dot product and scaled dot product attention mechanisms within the model.
2. **Model Enhancement**: Upgrading the model to include one-head self-attention (`self_attention_model.py`), allowing the GPT to focus on different parts of the input sequence.

## Step 6: Transitioning to Multi-Head Self-Attention
1. **Multi-Head Attention Class**: Construction of a new class to manage multi-head attention, facilitating the model to jointly attend to information from different representation subspaces.
2. **Concatenating Results**: Combining the outputs from the different attention heads to feed into subsequent layers.

## Step 7: Adding Feed-Forward Networks
1. **Feed-Forward Layer**: Inclusion of a fully connected feed-forward neural network using the ReLU activation function within a new `FeedForward` class.

## Step 8: Formulating Blocks
1. **Neural Network Blocks**: Assembling blocks of neural networks that replicate the transformer's blocks by repeating the multi-head self-attention and feed-forward network layers.

Each of these steps should be accompanied by the relevant code snippets and detailed explanations to provide clarity and guidance on how to implement each part of the model. Further steps will be detailed as the model development progresses.
