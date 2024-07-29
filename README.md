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

## Data Preprocessing
```
document, summary = utils.preprocess(train_data)
document_test, summary_test = utils.preprocess(test_data)

tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token, lower=False)
tokenizer.fit_on_texts(documents_and_summary)

inputs = tokenizer.texts_to_sequences(document)
targets = tokenizer.texts_to_sequences(summary)
```

## Training
```
# Initialize model
transformer = Transformer(num_layers, embedding_dim, num_heads, fully_connected_dim,
                          vocab_size, vocab_size, max_pos_encoding, max_pos_encoding)

# Training loop
for epoch in range(epochs):
    for batch, (inp, tar) in enumerate(dataset):
        train_step(transformer, inp, tar)
        
    print(f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
```

## Inference
```
def summarize(model, input_document):
    input_document = tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
    encoder_input = tf.expand_dims(input_document[0], 0)
    
    output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)
    
    for i in range(decoder_maxlen):
        predicted_id = next_word(model, encoder_input, output)
        output = tf.concat([output, predicted_id], axis=-1)
        
        if predicted_id == tokenizer.word_index["[EOS]"]:
            break

    return tokenizer.sequences_to_texts(output.numpy())[0]

# Generate summary
summary = summarize(transformer, input_document)
```

## Key Functions

- positional_encoding(positions, d_model): Generates positional encodings
- create_padding_mask(decoder_token_ids): Creates mask for padding tokens
- create_look_ahead_mask(sequence_length): Creates mask to prevent looking ahead
- scaled_dot_product_attention(q, k, v, mask): Computes attention weights
- EncoderLayer and DecoderLayer: Main components of the transformer
- Transformer: Combines encoder and decoder into full model
- train_step(model, inp, tar): Performs one training step
- next_word(model, encoder_input, output): Predicts the next word in the summary
- summarize(model, input_document): Generates a complete summary
