import pickle
import sys
import json
import numpy as np
valid_relations = { #made this into an actual dictionary
    ("anatomical location", "human"): ["located in"],
    ("anatomical location", "animal"):[ "located in"],
("bacteria", "bacteria"): ["interact"],
    ("bacteria", "chemical"): ["interact"],
    ("bacteria", "drug"): ["interact"],
    ("bacteria", "DDF"):["influence"],
    ("bacteria", "gene"): ["change expression"],
    ("bacteria", "human" ):["located in"],
    ("bacteria", "animal" ):["located in"],
    ("bacteria", "microbiome" ):["part of"],
    ("chemical", "anatomical location" ):["located in"],
    ("chemical", "human" ):["located in"],
    ("chemical", "animal" ):["located in"],
    ("chemical", "chemical"):["interact", "part of"],
    ("chemical", "microbiome" ):["impact","produced by"],
    ("chemical", "bacteria" ):["impact"],
    ("dietary supplement" ,"bacteria"): ["impact"],
    ("drug", "bacteria" ):["impact"],
    ("food", "bacteria" ):["impact"],
    ("chemical", "microbiome" ):["impact"],
    ("dietary supplement", "microbiome"): ["impact"],
    ("drug", "microbiome" ):["impact"],
    ("food", "microbiome" ):["impact"],
    ("chemical", "DDF" ):["influence"],
    ("dietary supplement", "DDF" ):["influence"],
    ("food", "DDF" ):["influence"],
    ("chemical", "gene" ):["change expression"],
    ("dietary supplement", "gene" ):["change expression"],
    ("drug", "gene" ):["change expression"],
    ("food", "gene" ):["change expression"],
    ("chemical", "human" ):["administered"],
    ("dietary supplement", "human" ):["administered"],
    ("drug", "human" ):["administered"],
    ("food", "human" ):["administered"],
    ("chemical", "animal" ):["administered"],
    ("dietary supplement", "animal"):["administered"],
    ("drug", "animal" ):["administered"],
    ("food", "animal" ):["administered"],
    ("DDF", "anatomical location" ):["strike"],
    ("DDF", "bacteria" ):["change abundance"],
    ("DDF", "microbiome" ):["change abundance"],
    ("DDF", "chemical" ):["interact"],
    ("DDF", "DDF" ):["affect", "is a"],
    ("DDF", "human" ):["target"],
    ("DDF", "animal" ):["target"],
    ("drug", "chemical" ):["interact"],
    ("drug", "drug" ):["interact"],
    ("drug", "DDF" ):["change effect"],
    ("human", "biomedical technique" ):["used by"],
    ("animal", "biomedical technique" ):["used by"],
    ("microbiome", "biomedical technique" ):["used by"],
    ("microbiome", "anatomical location" ):["located in"],
    ("microbiome", "human" ):["located in"],
    ("microbiome", "animal" ):["located in"],
    ("microbiome", "gene" ):["change expression"],
    ("microbiome", "DDF" ):["is linked to"],
    ("microbiome", "microbiome" ):["compared to"]
}

probs = {}
for i in range(1, len(sys.argv)-1):
    probs[sys.argv[i]] = pickle.load(open(sys.argv[i], "rb"))
flat_all_probs = {}
for p in probs:
    flat_all_probs[p] = [] 
    for batch in probs[p]:
        flat_all_probs[p].extend(batch)
flat_probs = np.array([flat_all_probs[p] for p in flat_all_probs])
best_weights_ensemble = pickle.load(open("best_weights_ensemble.pkl", "rb"))
ensemble_probs = np.zeros_like(flat_all_probs[0])
for i, probs in enumerate(flat_probs):
    ensemble_probs += best_weights_ensemble[i] * probs
flat_preds = np.argmax(ensemble_probs, axis=1)

testdata = None
with open(sys.argv[-1], "r") as file:
    testdata = json.load(file)
binary_tag_based_relations = {}
ternary_tag_based_relations = {}
ternary_mention_based_relations = {}
label_to_int = {'NONE': 0, 'impact': 1, 'influence': 2, 'interact': 3, 'located in': 4, 'change expression': 5, 'target': 6, 'part of': 7, 'used by': 8, 'change abundance': 9, 'is linked to': 10, 'strike': 11, 'affect': 12, 'change effect': 13, 'produced by': 14, 'administered': 15, 'is a': 16, 'compared to': 17}

counter = 0
for t in testdata:
    if t["doc_id"] not in binary_tag_based_relations:
        binary_tag_based_relations[t["doc_id"]] = {"binary_tag_based_relations": []}
        ternary_tag_based_relations[t["doc_id"]] = {"ternary_tag_based_relations":[]}
        ternary_mention_based_relations[t["doc_id"]] = {"ternary_mention_based_relations":[]}
    if flat_preds[counter]!=0 and list(label_to_int.keys())[flat_preds[counter]] in valid_relations[(t["subject_label"], t["object_label"])]:
        if  {"subject_label":t["subject_label"],"object_label":t["object_label"]} not in binary_tag_based_relations[t["doc_id"]]["binary_tag_based_relations"]:
            binary_tag_based_relations[t["doc_id"]]["binary_tag_based_relations"].append({"subject_label":t["subject_label"],"object_label":t["object_label"]})
        if {"subject_label":t["subject_label"],"predicate": list(label_to_int.keys())[flat_preds[counter]],"object_label":t["object_label"]} not in ternary_tag_based_relations[t["doc_id"]]["ternary_tag_based_relations"]:
            ternary_tag_based_relations[t["doc_id"]]["ternary_tag_based_relations"].append({"subject_label":t["subject_label"],"predicate": list(label_to_int.keys())[flat_preds[counter]],"object_label":t["object_label"]})
        if {"subject_text_span":t["subject"], "subject_label":t["subject_label"], "predicate":list(label_to_int.keys())[flat_preds[counter]], "object_text_span":t["object"], "object_label":t["object_label"]} not in ternary_mention_based_relations[t["doc_id"]]["ternary_mention_based_relations"]:
            ternary_mention_based_relations[t["doc_id"]]["ternary_mention_based_relations"].append({"subject_text_span":t["subject"], "subject_label":t["subject_label"], "predicate":list(label_to_int.keys())[flat_preds[counter]], "object_text_span":t["object"], "object_label":t["object_label"]})
    counter +=1

with open("binary_tag_based_relations.json", "w") as f:
    json.dump(binary_tag_based_relations, f, indent=4)
with open("ternary_tag_based_relations.json", "w") as f:
    json.dump(ternary_tag_based_relations, f, indent=4)
with open("ternary_mention_based_relations.json", "w") as f:
    json.dump(ternary_mention_based_relations, f, indent=4)
