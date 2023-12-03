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

## Step 8: Formulating Blocks (Nx Layer)
1. **Block Construction**: Designing a class for transformer blocks that can be stacked to form the `NX` layer, which adds depth and parallel processing capabilities to the model.
2. **Parallelization and Complexity**: Explanation of how stacking multiple transformer blocks (NX) can enhance the model's capacity to learn complex patterns.

## Step 9: Adding Residual Connections
1. **Introduction to Residual Connections**: Discussing the importance of residual connections in transformer models for facilitating deeper network training without the vanishing gradient problem.
2. **Implementation in Blocks**: Detailing the implementation of residual connections in both self-attention and feed-forward layers within the blocks to improve learning efficiency and model performance.

## Step 10: Incorporating Layer Normalization
1. **Understanding Layer Normalization**: Exploring the concept of layer normalization and its contrast with batch normalization in stabilizing the learning process.
2. **LayerNorm Class**: Creating a `LayerNorm` class with parameters epsilon, gamma, and beta to calculate and apply normalization on a layer-by-layer basis.
3. **Optimization via Normalization**: Discussing how adding layer normalization at various points in the model aids in optimization by normalizing the output of each layer before it passes to the next.

## Step 11: Implementing Dropout
1. **Purpose of Dropout**: Introducing dropout as a regularization technique to prevent overfitting by randomly deactivating a subset of neurons during training, thus forcing the model to learn more robust features.
2. **Application in the Model**: Detailed instructions on integrating dropout layers in the transformer blocks, particularly after attention and before feed-forward layers, to generalize learning and avoid over-reliance on specific features or patterns.

## Step 12: Scaling the Model
1. **Benefits of Scaling**: Explaining the motivation behind scaling up the model, which includes increased capacity for learning complex patterns and relationships in data.
2. **Increased Dimensions**: Guidance on increasing the batch size, block size, and dimensionality of the model, which involves more features per attention head and more layers.
3. **Deep Network Construction**: Discussing the implications of a deeper network with more parameters, and how this affects training time, resource requirements, and potential for capturing nuanced language structures.
4. **Considerations for Larger Models**: Providing considerations for hardware requirements and optimization techniques to efficiently train larger models without sacrificing performance or encountering hardware limitations.

