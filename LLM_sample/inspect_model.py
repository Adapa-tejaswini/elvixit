import torch

# Load the saved model
checkpoint = torch.load(
    "mini_llm.pth",
    map_location="cpu",
    weights_only=False
)
print("Keys inside mini_llm.pth:")
print(checkpoint.keys())
print("\Vocabulary size:")
print(len(checkpoint["vocabulary"]))
print("\Embedding weights shape:")
print(checkpoint["embedding"]["weight"].shape)
print("\Output layer weights shape:")
print(checkpoint["output_layer"]["weight"].shape)
print("\Output layer bias shape:")
print(checkpoint["output_layer"]["bias"].shape)
print("\Vocabulary:")
print(checkpoint["vocabulary"])
print("\Token IDs:")
print(checkpoint["token_to_id"])