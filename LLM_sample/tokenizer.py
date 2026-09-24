import re
# Open training data
file = open("data/training.txt", "r", encoding="utf-8")
text = file.read()
file.close()
# Split text into words and punctuation
tokens = re.findall(r"\w+|[^\w\s]", text)
# Remove duplicate words
vocabulary = sorted(set(tokens))
# Give each word a number
token_to_id = {}
for i in range(len(vocabulary)):
    token_to_id[vocabulary[i]] = i
# Print some information
print("Training text:")
print(text)
print("\Tokens:")
print(tokens)
print("\Vocabulary:")
print(vocabulary)
print("\Vocabulary size:")
print(len(vocabulary))
print("\Token IDs:")
for token in vocabulary:
    print(token, "->", token_to_id[token])