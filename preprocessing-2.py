import json
import sys
import random

# Valid relations dictionary
valid_relations = {
    ("Anatomical Location", "Human"): "Located in",
    ("Anatomical Location", "Animal"): "Located in",
    ("Bacteria", "Bacteria"): "Interact",
    ("Bacteria", "Chemical"): "Interact",
    ("Bacteria", "Drug"): "Interact",
    ("Bacteria", "DDF"): "Influence",
    ("Bacteria", "Gene"): "Change expression",
    ("Bacteria", "Human"): "Located in",
    ("Bacteria", "Animal"): "Located in",
    ("Bacteria", "Microbiome"): "Part of",
    ("Chemical", "Anatomical Location"): "Located in",
    ("Chemical", "Human"): "Located in",
    ("Chemical", "Animal"): "Located in",
    ("Chemical", "Chemical"): "Interact",
    ("Chemical", "Microbiome"): "Impact",
    ("Chemical", "Microbiome"): "Produced by",
    ("Chemical", "Bacteria"): "Impact",
    ("Dietary Supplement", "Bacteria"): "Impact",
    ("Drug", "Bacteria"): "Impact",
    ("Food", "Bacteria"): "Impact",
    ("Chemical", "Microbiome"): "Impact",
    ("Dietary Supplement", "Microbiome"): "Impact",
    ("Drug", "Microbiome"): "Impact",
    ("Food", "Microbiome"): "Impact",
    ("Chemical", "DDF"): "Influence",
    ("Dietary Supplement", "DDF"): "Influence",
    ("Drug", "DDF"): "Influence",
    ("Food", "DDF"): "Influence",
    ("Chemical", "Gene"): "Change expression",
    ("Dietary Supplement", "Gene"): "Change expression",
    ("Drug", "Gene"): "Change expression",
    ("Food", "Gene"): "Change expression",
    ("Chemical", "Human"): "Administered",
    ("Dietary Supplement", "Human"): "Administered",
    ("Drug", "Human"): "Administered",
    ("Food", "Human"): "Administered",
    ("Chemical", "Animal"): "Administered",
    ("Dietary Supplement", "Animal"): "Administered",
    ("Drug", "Animal"): "Administered",
    ("Food", "Animal"): "Administered",
    ("DDF", "Anatomical Location"): "Strike",
    ("DDF", "Bacteria"): "Change abundance",
    ("DDF", "Microbiome"): "Change abundance",
    ("DDF", "Chemical"): "Interact",
    ("DDF", "DDF"): "Affect",
    ("DDF", "DDF"): "Is a",
    ("DDF", "Human"): "Target",
    ("DDF", "Animal"): "Target",
    ("Drug", "Chemical"): "Interact",
    ("Drug", "DDF"): "Change effect",
    ("Human", "Biomedical Technique"): "Used by",
    ("Animal", "Biomedical Technique"): "Used by",
    ("Microbiome", "Biomedical Technique"): "Used by",
    ("Microbiome", "Anatomical Location"): "Located in",
    ("Microbiome", "Human"): "Located in",
    ("Microbiome", "Animal"): "Located in",
    ("Microbiome", "Gene"): "Change expression",
    ("Microbiome", "DDF"): "Is linked to",
    ("Microbiome", "Microbiome"): "Compared to"
}

# Specify the number of NONE relations to keep
NUM_NONE_RELATIONS = 1819  # Change this number as needed

# Load dataset
file_path = sys.argv[1]
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

def preprocess_data(data):
    processed_data = []
    none_relations = []  # Store NONE relations separately

    for doc_id, doc_data in data.items():
        metadata = doc_data.get("metadata", {})
        abstract = metadata.get("abstract", "")
        title = metadata.get("title", "")
        relations = doc_data.get("relations", [])
        entities = doc_data.get("entities", [])
        used = set()

        # Process valid relations
        for relation in relations:
            subject_start = relation.get("subject_start_idx")
            subject_end = relation.get("subject_end_idx")
            object_start = relation.get("object_start_idx")
            object_end = relation.get("object_end_idx")
            predicate = relation.get("predicate")
            subject_text = relation.get("subject_text_span")
            object_text = relation.get("object_text_span")
            subject_location = relation.get("subject_location")
            object_location = relation.get("object_location")
            relative_subject_start = subject_start
            relative_subject_end = subject_end
            relative_object_start = object_start
            relative_object_end = object_end

            if subject_location == "abstract":
                relative_subject_start += len(title)
                relative_subject_end += len(title)
            if object_location == "abstract":
                relative_object_start += len(title)
                relative_object_end += len(title)

            assert (title + abstract)[relative_subject_start:relative_subject_end + 1] == subject_text
            assert (title + abstract)[relative_object_start:relative_object_end + 1] == object_text

            processed_data.append({
                "sample": title + abstract,
                "subject": subject_text,
                "subject_label": relation.get("subject_label"),
                "object": object_text,
                "object_label": relation.get("object_label"),
                "relation": predicate,
                "relative_subject_start": relative_subject_start,
                "relative_subject_end": relative_subject_end,
                "relative_object_start": relative_object_start,
                "relative_object_end": relative_object_end,
                "title_length": len(title)
            })
            used.add(((relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)))

        # Process "NONE" relations
        for e in entities:
            for e1 in entities:
                if e != e1:
                    relative_subject_start = e.get("start_idx")
                    relative_subject_end = e.get("end_idx")
                    relative_object_start = e1.get("start_idx")
                    relative_object_end = e1.get("end_idx")
                    subject_location = e.get("location")
                    object_location = e1.get("location")

                    if subject_location == "abstract":
                        relative_subject_start += len(title)
                        relative_subject_end += len(title)
                    if object_location == "abstract":
                        relative_object_start += len(title)
                        relative_object_end += len(title)

                    # Check if it’s a "NONE" relation
                    if ((relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)) not in used:
                        subject_type = e.get("label", "")
                        object_type = e1.get("label", "")

                        # Only consider "NONE" if it's a valid subject-object pair
                        if (subject_type, object_type) in valid_relations:
                            none_relations.append({
                                "sample": title + abstract,
                                "subject": e["text_span"],
                                "subject_label": subject_type,
                                "object": e1["text_span"],
                                "object_label": object_type,
                                "relation": "NONE",
                                "relative_subject_start": relative_subject_start,
                                "relative_subject_end": relative_subject_end,
                                "relative_object_start": relative_object_start,
                                "relative_object_end": relative_object_end,
                                "title_length": len(title)
                            })

    # Randomly sample NONE relations up to the specified limit
    sampled_none_relations = random.sample(none_relations, min(NUM_NONE_RELATIONS, len(none_relations)))

    # Combine the valid and sampled NONE relations
    final_data = processed_data + sampled_none_relations

    return final_data

# Process the dataset
processed_data = preprocess_data(data)

# Output processed data (modify as needed)
output_path = "processed_data.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=4)
    
print(f"Processed data saved to {output_path}")
