import re
# Read training data
file = open("data/training.txt", "r", encoding="utf-8")
text = file.read()
file.close()
# Convert text into tokens
tokens = re.findall(r"\w+|[^\w\s]", text)
# Create vocabulary
vocabulary = sorted(set(tokens))
# Create token IDs
token_to_id = {}
for i in range(len(vocabulary)):
    token_to_id[vocabulary[i]] = i
# Convert all tokens into numbers
token_ids = []
for token in tokens:
    token_ids.append(token_to_id[token])
print("Tokens:")
print(tokens)
print("\Token IDs:")
print(token_ids)
# Create input and target pairs
print("\Training pairs:")
for i in range(len(token_ids) - 1):
    input_token = token_ids[i]
    target_token = token_ids[i + 1]
    print(input_token, "->", target_token)