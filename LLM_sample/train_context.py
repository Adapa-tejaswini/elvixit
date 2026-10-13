import re
import torch
import torch.nn as nn

#  Read training data

file = open(
    "data/training.txt",
    "r",
    encoding="utf-8"
)
text = file.read()
file.close()
# Create tokens


tokens = []
lines = text.splitlines()
for line in lines:
    line_tokens = re.findall(
        r"\w+|[^\w\s]",
        line
    )
    for token in line_tokens:
        tokens.append(token)
    # End of each sentence
    tokens.append("<END>")

# Create vocabulary
vocabulary = sorted(set(tokens))

# Create token IDs


token_to_id = {}

for i in range(len(vocabulary)):

    token_to_id[vocabulary[i]] = i

# Convert tokens to IDs


token_ids = []

for token in tokens:

    token_ids.append(
        token_to_id[token]
    )
print("Vocabulary size:", len(vocabulary))

print("Total tokens:", len(token_ids))

# Create context data


context_size = 5

inputs = []

targets = []


for i in range(
    len(token_ids) - context_size
):

    input_sequence = token_ids[
        i:i + context_size
    ]

    target_token = token_ids[
        i + context_size
    ]

    inputs.append(
        input_sequence
    )

    targets.append(
        target_token
    )

# Convert to tensors


inputs = torch.tensor(
    inputs
)
targets = torch.tensor(
    targets
)
print("\Input shape:")
print(inputs.shape)
print("\Target shape:")
print(targets.shape)
print("\First input:")
print(inputs[0])
print("\First target:")

print(targets[0])

# Model settings


vocab_size = len(vocabulary)

embedding_size = 32


# Create layers


embedding = nn.Embedding(
    vocab_size,
    embedding_size
)


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


output_layer = nn.Linear(
    embedding_size,
    vocab_size
)

# Loss function


loss_function = nn.CrossEntropyLoss()

# Optimizer


optimizer = torch.optim.Adam(
    list(embedding.parameters())
    + list(query_layer.parameters())
    + list(key_layer.parameters())
    + list(value_layer.parameters())
    + list(output_layer.parameters()),
    lr=0.005
)

# Causal mask


mask = torch.triu(
    torch.ones(
        context_size,
        context_size
    ),
    diagonal=1
).bool()

#  Training


epochs = 1500

for epoch in range(epochs):
    # Convert IDs to embeddings
   
    embedded = embedding(
        inputs
    )
    # Query
 
    query = query_layer(
        embedded
    )
    # Key
    key = key_layer(
        embedded
    )

    # Value
    value = value_layer(
        embedded
    )
    # Attention scores
    scores = torch.matmul(
        query,
        key.transpose(1, 2)
    )
    # Scale scores
    scores = scores / (
        embedding_size ** 0.5
    )
    # Apply causal mask
    scores = scores.masked_fill(
        mask,
        float("-inf")
    )
    # Attention weights
    attention_weights = torch.softmax(
        scores,
        dim=-1
    )
    # Context

    context = torch.matmul(
        attention_weights,
        value
    )
    # Use only the LAST position
    last_context = context[:, -1, :]
    # Predict next word
    predictions = output_layer(
        last_context
    )
    # Calculate loss
    loss = loss_function(
        predictions,
        targets
    )
    # Clear gradients
    optimizer.zero_grad()
    # Calculate gradients
    loss.backward()
    # Update model
    optimizer.step()
    # Print loss
    if (epoch + 1) % 100 == 0:

        print(
            "Epoch:",
            epoch + 1,
            "Loss:",
            loss.item()
        )

#Save model


torch.save(
    {
        "embedding": embedding.state_dict(),

        "query_layer": query_layer.state_dict(),

        "key_layer": key_layer.state_dict(),

        "value_layer": value_layer.state_dict(),

        "output_layer": output_layer.state_dict(),

        "token_to_id": token_to_id,

        "vocabulary": vocabulary,

        "context_size": context_size,

        "embedding_size": embedding_size
    },
    "mini_transformer_context.pth"
)


print("\Context Transformer trained successfully!")
print("Model saved as:")
print("mini_transformer_context.pth")