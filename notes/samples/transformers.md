# Transformers

## Overview
The transformer architecture was introduced in the 2017 paper "Attention Is All You Need". It replaced recurrence with attention, so all tokens in a sequence can be processed in parallel.

## Self-attention
Self-attention lets every token look at every other token in the sequence. Each token produces a query, a key and a value vector. Attention weights come from the softmax of query-key dot products, scaled by the square root of the key dimension.

## Multi-head attention
Multi-head attention runs several attention operations in parallel, each with its own learned projections. Different heads can focus on different relationships, such as grammar or meaning. Their outputs are concatenated and projected back.

## Positional encoding
Attention has no built-in sense of word order, so positional encodings are added to the token embeddings to tell the model where each token is.

## Encoder and decoder
BERT uses only the encoder and is good at understanding text. GPT models use only the decoder and generate text one token at a time.