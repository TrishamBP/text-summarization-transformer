# Transformer-Based Text Summarization

## Overview

This project implements a transformer model for text summarization using TensorFlow, inspired by Google's seminal 2017 paper "Attention Is All You Need". The model uses an encoder-decoder architecture with self-attention mechanisms to generate abstractive summaries of input text.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Key Components](#key-components)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Training](#training)
6. [Inference](#inference)
7. [Key Functions](#key-functions)
8. [Custom Learning Rate Scheduler](#custom-learning-rate-scheduler)

## Model Architecture

The transformer model consists of:

- Encoder: Processes the input sequence
![image](https://github.com/user-attachments/assets/87acfcda-2b50-42cb-9692-57bbfe8671e0)
- Decoder: Generates the output summary
![image](https://github.com/user-attachments/assets/17d1ceda-a370-46de-87c0-5e29ac77bf23)
- Multi-head attention layers: Allow the model to focus on different parts of the input
- Feed-forward neural networks: Process the attention output
- Layer normalization: Stabilizes the learning process
![image](https://github.com/user-attachments/assets/df24b9eb-2478-4ee3-a79b-54cbcfa3ab34)


## Key Components

- Positional encoding to maintain sequence order information
- Masked self-attention in the decoder to prevent looking ahead
- Scaled dot-product attention as the core attention mechanism
