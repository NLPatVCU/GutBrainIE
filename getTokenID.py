from transformers import DebertaV2TokenizerFast

# Load the DeBERTa tokenizer
tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")

# Input sentence
sentence = "the gut microbiome communicates with the brain"

# Tokenize
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print the result
print(f"Sentence: {sentence}")
print("Tokens and their IDs:")
for token, token_id in zip(tokens, token_ids):
    print(f"  {token:<15} --> {token_id}")
