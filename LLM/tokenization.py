text="I am learning about large language Models" # orginal message
tokens=text.split() # split the sentence into words(or) tokens
vocabulary={} 
for token in tokens: # going through each token in the senetence
  if token not in vocabulary: # check if token is alredy there in vocabulary dict
    vocabulary[token]=len(vocabulary) # assign a unique number to each token
print("tokens: ",tokens)
print("vocabulary:",vocabulary)
token_ids=[]
for token in tokens:
  token_ids.append(vocabulary[token])
print("Token Ids:",token_ids)
