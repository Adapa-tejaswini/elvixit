import torch
import torch.nn as nn
checkpoint = torch.load(
    "mini_transformer_context.pth",
    map_location="cpu",
    weights_only=False
)#load training data
token_to_id = checkpoint["token_to_id"]
vocabulary = checkpoint["vocabulary"]
context_size = checkpoint["context_size"]
embedding_size = checkpoint["embedding_size"]
vocab_size = len(vocabulary)
# Create model layers
embedding = nn.Embedding(vocab_size, embedding_size)
query_layer = nn.Linear(embedding_size, embedding_size)
key_layer = nn.Linear(embedding_size, embedding_size)
value_layer = nn.Linear(embedding_size, embedding_size)
output_layer = nn.Linear(embedding_size, vocab_size)
# Load trained weights
embedding.load_state_dict(checkpoint["embedding"])
query_layer.load_state_dict(checkpoint["query_layer"])
key_layer.load_state_dict(checkpoint["key_layer"])
value_layer.load_state_dict(checkpoint["value_layer"])
output_layer.load_state_dict(checkpoint["output_layer"])
# Put model in evaluation mode
embedding.eval()
query_layer.eval()
key_layer.eval()
value_layer.eval()
output_layer.eval()
start_text = "Artificial intelligence is a field"
words = start_text.split()
context_ids = []
for word in words:
    context_ids.append(token_to_id[word])
print("Generated text:", start_text, end="")
for step in range(20):
    current_context = context_ids[-context_size:]
    input_tensor = torch.tensor([current_context])
    # Convert tokens to embeddings
    embedded = embedding(input_tensor)
    # Create Query, Key and Value
    query = query_layer(embedded)
    key = key_layer(embedded)
    value = value_layer(embedded)
    # Calculate attention scores
    scores = torch.matmul(
        query,
        key.transpose(1, 2)
    )
    # Scale attention scores
    scores = scores / (embedding_size ** 0.5)
    # Causal mask
    current_size = len(current_context)
    mask = torch.triu(
        torch.ones(current_size, current_size),
        diagonal=1
    ).bool()
    scores = scores.masked_fill(
        mask,
        float("-inf")
    )
    # Convert scores into attention weights
    attention_weights = torch.softmax(
        scores,
        dim=-1
    )
    # Create context-aware representation
    context = torch.matmul(
        attention_weights,
        value
    )
    # Use the last token representation
    last_context = context[:, -1, :]
    # Predict next token
    prediction = output_layer(last_context)
    next_token_id = torch.argmax(
        prediction,
        dim=-1
    ).item()
    # Convert ID back to word
    next_token = vocabulary[next_token_id]
    # Stop at end of sentence
    if next_token == "<END>":
        break
    # Print generated word
    print(" " + next_token, end="")
    # Add token to context
    context_ids.append(next_token_id)
print()