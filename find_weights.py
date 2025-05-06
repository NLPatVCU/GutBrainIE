import pickle
import sys
import json
from evaluate import eval_submission_6_4_ternary_mention_RE
import optuna
reg_probs_path = sys.argv[1]
cnn_probs_path = sys.argv[2]

label_to_int = {'NONE': 0, 'impact': 1, 'influence': 2, 'interact': 3, 'located in': 4, 'change expression': 5, 'target': 6, 'part of': 7, 'used by': 8, 'change abundance': 9, 'is linked to': 10, 'strike': 11, 'affect': 12, 'change effect': 13, 'produced by': 14, 'administered': 15, 'is a': 16, 'compared to': 17}

cnn_probs = pickle.load(open(cnn_probs_path, "rb"))
reg_probs = pickle.load(open(reg_probs_path, "rb"))


cnn_flat_probs = []
reg_flat_probs = []

for batch in cnn_probs:
    cnn_flat_probs.extend(batch)
for batch in reg_probs:
    reg_flat_probs.extend(batch)
all_probs = [reg_flat_probs, cnn_flat_probs]

def objective(trial):
    weights = [trial.suggest_float(f"weight_model_{i}", 0.0, 1.0) for i in range(len(all_probs))]
    weights = np.array(weights) / np.sum(weights)
    ensemble_probs = np.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        ensemble_probs += weights[i]*probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)


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

    testdata = None
    with open(sys.argv[3], "r") as file:
        testdata = json.load(file)
    binary_tag_based_relations = {}
    ternary_tag_based_relations = {}
    ternary_mention_based_relations = {}

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
    
    precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_6_4_ternary_mention_RE("ternary_mention_based_relations.json")

    return -micro_f1
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
best_weights = np.array([best_params[f'weight_model_{i}'] for i in range(len(model_probs))])
best_weights = best_weights / np.sum(best_weights)

best_f1 = -study.best_value

print(best_weights)
print(best_f1)


pickle.dump(best_weights, open("best_weights_ensemble.pkl", "wb"))
