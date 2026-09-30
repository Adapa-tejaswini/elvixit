import torch
import torch.nn as nn
# Vocabulary contains 58 tokens
vocab_size = 58
# Each token will be represented by 16 numbers
embedding_size = 16
# Create embedding layer
embedding = nn.Embedding(vocab_size, embedding_size)
# Create output layer
output_layer = nn.Linear(embedding_size, vocab_size)
# Example token
token = torch.tensor([2])
# Convert token ID into embedding
embedded_token = embedding(token)
print("Token ID:")
print(token)
print("\Embedding:")
print(embedded_token)
print("\Embedding shape:")
print(embedded_token.shape)
# Pass embedding through neural network
prediction = output_layer(embedded_token)
print("\Prediction:")
print(prediction)
print("\Prediction shape:")
print(prediction.shape)