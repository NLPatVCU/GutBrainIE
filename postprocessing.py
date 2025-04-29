import pickle
import sys
import json

preds = None
with open(sys.argv[1], "rb") as f:
    preds = pickle.load(f)
flat_preds = []
for p in preds:
    flat_preds.extend(p.tolist())
testdata = None
with open(sys.argv[2], "r") as file:
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
    if flat_preds[counter]!=0:
        if  {"subject_label":t["subject_label"],"object_label":t["object_label"]} not in binary_tag_based_relations[t["doc_id"]]["binary_tag_based_relations"]:
            binary_tag_based_relations[t["doc_id"]]["binary_tag_based_relations"].append({"subject_label":t["subject_label"],"object_label":t["object_label"]})
        if {"subject_label":t["subject_label"],"predicate": list(label_to_int.keys())[flat_preds[counter]],"object_label":t["object_label"]} not in ternary_tag_based_relations[t["doc_id"]]["ternary_tag_based_relations"]:
            ternary_tag_based_relations[t["doc_id"]]["ternary_tag_based_relations"].append({"subject_label":t["subject_label"],"predicate": list(label_to_int.keys())[flat_preds[counter]],"object_label":t["object_label"]})
        ternary_mention_based_relations[t["doc_id"]]["ternary_mention_based_relations"].append({"subject_text_span":t["subject"], "subject_label":t["subject_label"], "predicate":list(label_to_int.keys())[flat_preds[counter]], "object_text_span":t["object"], "object_label":t["object_label"]})
    counter +=1
with open("binary_tag_based_relations.json", "w") as f:
    json.dump(binary_tag_based_relations, f, indent=4)
with open("ternary_tag_based_relations.json", "w") as f:
    json.dump(ternary_tag_based_relations, f, indent=4)
with open("ternary_mention_based_relations.json", "w") as f:
    json.dump(ternary_mention_based_relations, f, indent=4)
