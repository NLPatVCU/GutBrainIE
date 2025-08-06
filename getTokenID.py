from transformers import DebertaV2TokenizerFast
import json


tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")

with open("trainData.json", "r") as f:
    data = json.load(f)
sample = data[0]["sample"]

encoding = tokenizer(
    sample,
    return_offsets_mapping=True,  
    return_tensors=None,
    add_special_tokens=True,
    truncation=True,
    max_length=350,
    padding='max_length'
)

input_ids = encoding["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(f"\n{'Token ID':<10} {'Token':<20}")
print("=" * 30)
for tid, tok in zip(input_ids, tokens):
    print(f"{tid:<10} {tok:<20}")
