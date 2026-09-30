import torch
import torch.nn as nn
# Vocabulary
vocab_size = 58
# Each token becomes 16 numbers
embedding_size = 16
#Create token embeddings
embedding = nn.Embedding(
    vocab_size,
    embedding_size
)
#Create self-attention
query_layer = nn.Linear(
    embedding_size,
    embedding_size
)

key_layer = nn.Linear(
    embedding_size,
    embedding_size
)

value_layer = nn.Linear(
    embedding_size,
    embedding_size
)

#Create output layer

output_layer = nn.Linear(
    embedding_size,
    vocab_size
)
#Give the model some tokens
tokens = torch.tensor([
    [2, 28, 29]
])
print("Token IDs:")
print(tokens)
print("\Token shape:")
print(tokens.shape)
# Convert tokens to embeddings
embedded = embedding(tokens)
print("\Embeddings:")
print(embedded)
print("\Embedding shape:")
print(embedded.shape)
# Create Query, Key and Value
query = query_layer(embedded)
key = key_layer(embedded)
value = value_layer(embedded)
print("\Query shape:")
print(query.shape)
print("\Key shape:")
print(key.shape)
print("\Value shape:")
print(value.shape)

#Calculate attention scores

scores = torch.matmul(
    query,
    key.transpose(1, 2)
)
print("\Attention scores:")
print(scores)
print("\Attention score shape:")
print(scores.shape)
#Convert scores to weights
attention_weights = torch.softmax(
    scores,
    dim=-1
)
print("\Attention weights:")
print(attention_weights)
# Create context
context = torch.matmul(
    attention_weights,
    value
)
print("\Context:")
print(context)
print("\Context shape:")
print(context.shape)
#Predict next tokens
prediction = output_layer(context)
print("\Prediction:")
print(prediction)
print("\Prediction shape:")
print(prediction.shape)

#Find predicted token
predicted_token = torch.argmax(
    prediction,
    dim=-1
)
print("\Predicted token ID:")
print(predicted_token)