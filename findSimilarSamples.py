from transformers import DebertaV2TokenizerFast, AutoModel
from sentence_transformers import util
import torch
import json
import sys

# Load tokenizer and model 
tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
model.eval()

def getSamplesAndRelations(file):
    with open(file, 'r') as f:
        data = json.load(f)
    relations = {'administered': [], 'change effect': [], 'is a': [], 'compared to': []}
    for item in data:
        if item['relation'] in relations:
            relations[item['relation']].append(item['sample'])
    return relations

def getData(file):
    with open(file, 'r') as f:
        return json.load(f)

def getVectorRep(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  

def getAvgVectorRep(sentences):
    vecs = [getVectorRep(sent) for sent in sentences]
    stacked = torch.stack(vecs)
    return stacked.mean(dim=0)

def compareVectorRep(v1, v2):
    return util.cos_sim(v1, v2)[0][0].item()

def outputToJSON(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# === MAIN ===
originalFile = sys.argv[1]
comparisonFile = sys.argv[2]
outputFile = sys.argv[3]

original_samples_by_relation = getSamplesAndRelations(originalFile)
comparison_data = getData(comparisonFile)

results = []

for relation, original_samples in original_samples_by_relation.items():
    if not original_samples:
        continue
    avg_vec = getAvgVectorRep(original_samples)
    for item in comparison_data:
        if item['relation'] != relation:
            continue
        sample_vec = getVectorRep(item['sample'])
        sim = compareVectorRep(avg_vec, sample_vec)
        if sim >= 0.75:
            #item['similarity'] = sim --> use if you want to see similarity
            results.append(item)

outputToJSON(results, outputFile)

