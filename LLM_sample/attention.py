import torch
#  Create 3 token embeddings
tokens = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])
print("Token embeddings:")
print(tokens)
print("\Shape:")
print(tokens.shape)
# Create Query, Key, Value
query = tokens
key = tokens
value = tokens
print("\Query:")
print(query)
print("\Key:")
print(key)
print("\Value:")
print(value)
#Calculate attention scores
scores = torch.matmul(query, key.T)
print("\Attention scores:")
print(scores)
#Convert scores to probabilities
attention_weights = torch.softmax(scores, dim=1)
print("\Attention weights:")
print(attention_weights)
#Create new context-aware embedding
context = torch.matmul(attention_weights, value)
print("\Context-aware embeddings:")
print(context)
print("\Context shape:")
print(context.shape)