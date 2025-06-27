# soc_2025
LLM From scratch
# GPT Model Implementation: 4-Week Learning Journey

This repository contains Jupyter notebooks (`week2.ipynb`, `week3.ipynb`, `week4.ipynb`) documenting a 4-week project to implement a GPT-like model from scratch using Python and PyTorch. Week 1 focused on Python basics (no practice files due to prior familiarity), while Weeks 2, 3, and 4 cover tokenization, attention mechanisms, and the full GPT architecture for text generation.

## Project Overview

The goal is to build a GPT model for text generation, starting from text preprocessing to a complete transformer-based model. The notebooks cover tokenization (`week2.ipynb`), attention mechanisms (`week3.ipynb`), and the GPT model with text generation (`week4.ipynb`).

## Weekly Breakdown

### Week 1: Python Basics
- **Focus**: Covered foundational Python programming (e.g., variables, loops, functions).
- **Note**: No practice files were created due to prior familiarity with Python.

### Week 2: Tokenization and Data Preparation
- **Notebook**: `week2.ipynb`
- **Concepts Implemented**:
  - **Text Preprocessing**: Loaded a 20,479-character short story ("The Verdict" by Edith Wharton) for tokenization.
  - **Tokenization**: Split text into tokens (words, punctuation) using regular expressions (`re.split`) to handle whitespace, commas, periods, and special characters (e.g., `--`, `?`, `"`).
    - **SimpleTokenizerV1**: Converts text to token IDs using a vocabulary of 1,130 unique tokens, with `encode` (text to IDs) and `decode` (IDs to text) methods. Handles known tokens but fails on unknown words (e.g., "Hello").
    - **SimpleTokenizerV2**: Extends V1 by adding `<|unk|>` (unknown words) and `<|endoftext|>` (text boundary) tokens, increasing vocabulary to 1,132. Handles out-of-vocabulary words by replacing them with `<|unk|>`.
    - **Byte Pair Encoding (BPE)**: Introduced `tiktoken` (GPT-2's tokenizer) with a 50,257-token vocabulary, breaking unknown words into subword units (e.g., "someunknownPlace" tokenized correctly without `<|unk|>`).
  - **Data Loader**: Implemented `GPTDatasetV1` and `create_dataloader_v1` using PyTorch's `Dataset` and `DataLoader` for efficient batch processing.
    - **Sliding Window**: Creates input-target pairs with a context size (e.g., 4 tokens) and stride (e.g., 1 or 4) for next-word prediction tasks.
    - **Input-Target Pairs**: Inputs (`x`) are token sequences; targets (`y`) are sequences shifted by one token (e.g., `x=[290, 4920, 2241, 287]`, `y=[4920, 2241, 287, 257]`).
  - **Token Embeddings**: Converted token IDs to dense vectors (e.g., 256-dimensional) using `torch.nn.Embedding`. Example: 4 tokens with IDs `[2, 3, 5, 1]` mapped to a 4x3 tensor for a small vocabulary.
  - **Positional Embeddings**: Added position-specific vectors to token embeddings to encode token order, maintaining the same dimensionality (e.g., 256).
- **Key Classes**:
  - `SimpleTokenizerV1` & `SimpleTokenizerV2`: Basic and enhanced tokenizers.
  - `GPTDatasetV1`: Creates input-target pairs using a sliding window.
- **Outcome**: A robust pipeline for tokenizing text, handling unknown words, and preparing data for LLM training with embeddings.

### Week 3: Attention Mechanism Implementation
- **Notebook**: `week3.ipynb`
- **Concepts Implemented**:
  - **Attention Mechanism**: Computes relevance scores between tokens using **query**, **key**, and **value** matrices to focus on important parts of the input sequence.
    - **Key, Query, Value Matrices**: Represent tokens as vectors to calculate attention scores. Queries and keys compute similarity, while values provide the output.
    - **Dot Product**: Measures similarity between query and key vectors to determine attention weights.
    - **Softmax**: Normalizes attention scores into a probability distribution, ensuring weights sum to 1.
  - **Causal Attention**: Ensures tokens only attend to previous tokens (via masking), critical for autoregressive text generation.
  - **Multi-Head Attention**: Splits attention into multiple parallel "heads" to capture diverse relationships, improving model expressiveness.
  - **Weight Splits**: Optimizes attention by dividing weights across heads for efficient computation.
- **Key Classes**:
  - `CausalAttention`: Implements single-head attention with masking.
  - `MultiHeadAttentionWrapper` & `MultiHeadAttention`: Combine multiple attention heads for richer representations.
- **Outcome**: A functional attention mechanism for processing token sequences, foundational for transformer models.

### Week 4: GPT Model Architecture and Text Generation
- **Notebook**: `week4.ipynb`
- **Concepts Implemented**:
  - **GPT Architecture**: A transformer-based model with token and positional embeddings, transformer blocks, layer normalization, and a linear output head.
    - **Token & Positional Embeddings**: Convert token IDs to dense vectors and add positional information to capture sequence order.
    - **Transformer Block**: Combines multi-head attention, feedforward neural networks, and shortcut connections for deep learning.
    - **Layer Normalization**: Normalizes inputs to have zero mean and unit variance, stabilizing training. Uses trainable scale and shift parameters.
    - **Feedforward Network with GELU**: Applies a non-linear activation (Gaussian Error Linear Unit) for enhanced modeling, replacing simpler ReLU.
    - **Shortcut Connections**: Mitigate vanishing gradients by adding input to output, ensuring stable gradient flow across layers.
    - **Weight Tying**: Shares weights between token embedding and output layers to reduce parameters (from 163M to 124M).
  - **Text Generation**:
    - **Greedy Decoding**: Selects the most likely next token using `torch.argmax` on logits (optionally softmax-normalized).
    - **Context Management**: Crops input context to fit the model's context length (e.g., 1,024 tokens).
  - **Vanishing Gradient Problem**: Demonstrated via a deep neural network example, showing how shortcut connections prevent gradient diminishment.
- **Key Classes**:
  - `LayerNorm`: Normalizes inputs with trainable parameters.
  - `GELU`: Implements the GELU activation function.
  - `TransformerBlock`: Combines attention, feedforward, and normalization with shortcuts.
  - `GPTModel`: Full model with embeddings, transformer blocks, and output head.
  - `generate_text_simple`: Generates text by iteratively predicting tokens.
- **Outcome**: A 124M-parameter GPT model (with weight tying) that processes tokenized input and generates text (though untrained, producing gibberish).

## Key Technical Details
- **Configuration**: Uses `GPT_CONFIG_124M` with 50,257 vocab size, 1,024 context length, 768 embedding dimension, 12 heads, and 12 layers.
- **Tokenization**: Employs `tiktoken` with GPT-2 encoding for converting text to token IDs.
- **Model Size**: 163M parameters (621.83 MB), reduced to 124M with weight tying.
- **Dependencies**: Python, PyTorch, `tiktoken`, `matplotlib` (for visualizations).

## Why It Matters
- **Tokenization**: Converts raw text into numerical inputs suitable for LLMs, with BPE handling unknown words efficiently.
- **Attention Mechanism**: Enables the model to focus on relevant tokens, crucial for understanding context in language tasks.
- **Layer Normalization & Shortcuts**: Enhance training stability and prevent gradient issues in deep networks.
- **GELU Activation**: Provides smoother non-linearity compared to ReLU, improving model performance.
- **Text Generation**: Demonstrates the end-to-end process of converting token indices to meaningful (or potentially meaningful, post-training) text.
